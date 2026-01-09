from .utils import default_to_self
import pandas as pd
import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt
import re
import pyranges as pr
from typing import List, Tuple, Optional, Union


class meta_a2g:
    """
    Meta-transcript analysis of A→G RNA editing signals.

    This class implements a transcript-centric (meta-transcript) framework to
    aggregate per-base A→G editing statistics across genes by mapping genomic
    coordinates onto normalized transcript coordinates.

    The typical use case is to study the average A→G editing profile along
    transcripts (5' → 3') by:
        1. selecting one representative transcript per gene
        2. building an exon-level transcript coordinate system
        3. mapping per-base A→G signals onto normalized transcript positions
        4. binning signals across transcripts
        5. plotting the aggregated meta profile

    Coordinate System
    -----------------
    - Input genomic coordinates:
        - 1-based, inclusive (as in GTF and per-base TSV outputs)
    - Internal processing:
        - Exon-level intervals mapped to transcript coordinates
    - Output coordinates:
        - Normalized transcript position in [0, 1]

    Data Requirements
    -----------------
    1. Per-base A→G statistics table
       Required columns:
        - chrom
        - position          (1-based)
        - strand
        - alt_ratio
        - chrom_site_reads_count

    2. GTF-like annotation table
       Required columns:
        - chrom
        - start
        - end
        - strand
        - feature           (e.g. exon)
        - attr              (must contain gene_id, transcript_id, transcript_type)

    Only transcripts annotated as ``protein_coding`` are retained.

    Representative Transcript Selection
    -----------------------------------
    One transcript is selected per gene according to the following priority:
        1. MANE_Select
        2. transcripts tagged as "basic"
        3. transcript with the largest total exon length

    Transcripts with insufficient coverage can be filtered by specifying
    ``min_depth`` during mapping.

    Typical Workflow
    ----------------
    >>> meta = meta_a2g(
    ...     df_A2G_per_base=df_bases,
    ...     df_annotation=df_gtf,
    ...     exclude_chrom=("chrM",),
    ... )
    >>> meta.select_repr_rnas() \\
    ...     .build_exon_index() \\
    ...     .map_A2G_to_meta(min_depth=10) \\
    ...     .bin_meta_A2G(n_bins=100, agg="mean")
    >>> meta.plot_ax(
    ...     ylabel="A→G ratio",
    ...     title="Meta A→G",
    ... )

    Multiple Conditions
    -------------------
    Multiple meta-profiles can be plotted on the same axes by reusing ``plot_ax``
    with ``show=False`` and passing a shared matplotlib Axes.

    Notes
    -----
    - Mapping from genome to transcript coordinates is exon-aware and strand-aware.
    - All operations are vectorized and PyRanges-based for performance.
    - This class does not perform variant calling; it assumes precomputed
      per-base A→G statistics.

    Attributes
    ----------
    df_bases : pandas.DataFrame
        Per-base A→G statistics.
    df_annotation : pandas.DataFrame
        GTF-like gene annotation table.
    df_exon_index : pandas.DataFrame | None
        Exon-level transcript index for coordinate mapping.
    df_meta : pandas.DataFrame | None
        Per-base A→G signals mapped to normalized transcript coordinates.
    df_meta_plot : pandas.DataFrame | None
        Binned meta-transcript A→G profile for plotting.
    exclude_chrom : list | tuple | None
        Chromosomes to exclude from analysis.
    """

    def __init__(
        self,
        # ref_fasta,
        df_A2G_per_base,
        df_annotation,
        exclude_chrom: Optional[Union[list, tuple]] = None,
    ):
        """
        Initialize a meta-transcript A→G analysis object.

        Parameters
        ----------
        df_A2G_per_base : pandas.DataFrame
            Per-base A→G statistics table. Required columns:
            - chrom
            - position (1-based)
            - strand
            - alt_ratio
            - chrom_site_reads_count

        df_annotation : pandas.DataFrame
            GTF-like annotation table containing exon definitions.
            Required columns:
            - chrom
            - start
            - end
            - strand
            - feature
            - attr

        exclude_chrom : list | tuple | None, optional
            Chromosome names to exclude from all analyses
            (e.g. ("chrM",)), by default None.

        Notes
        -----
        This constructor does not perform any computation.
        Downstream analysis requires calling, in order:
            - select_repr_rnas()
            - build_exon_index()
            - map_A2G_to_meta()
        """
        # self.ref_fasta = ref_fasta
        self.df_bases = df_A2G_per_base
        self.exclude_chrom = exclude_chrom
        self.df_annotation = df_annotation
        self.df_exon_index = None
        self.df_meta = None
        self.df_meta_plot = None

    @default_to_self(("df_annotation",))
    def select_repr_rnas(
        self,
        df_annotation: Optional[pd.DataFrame] = None,
        *,
        prefer_mane: bool = True,
        prefer_basic: bool = True,
    ):
        """
        Select one representative transcript per gene.

        For each gene, exactly one transcript is retained according to
        the following priority rules:

        Priority
        --------
        1. MANE_Select transcript
        2. Transcripts tagged as "basic"
        3. Transcript with the largest total exon length

        Only transcripts annotated as ***protein_coding*** are considered.

        Parameters
        ----------
        df_annotation : pandas.DataFrame | None, optional
            GTF-like annotation table. If ``None``, uses ``self.df_annotation``.

        prefer_mane : bool, optional
            Whether to prioritize MANE_Select transcripts, by default True.

        prefer_basic : bool, optional
            Whether to prioritize transcripts tagged as "basic",
            by default True.

        Returns
        -------
        self.df_annotation : pandas.DataFrame
            Containing only the selected representative transcripts.
        """

        # -------- 0. 列裁剪 + 显式 copy（关键） --------
        df = df_annotation.loc[
            :, ["chrom", "start", "end", "strand", "feature", "attr"]
        ].copy()

        # -------- 1. 染色体 & 基因过滤 --------
        if self.exclude_chrom is not None:
            df = df.loc[~df.chrom.isin(self.exclude_chrom)]
        df = df.loc[
            df["attr"].str.contains('transcript_type "protein_coding"', na=False)
        ]

        # -------- 2. 向量化解析 gene_id / transcript_id（避免 apply） --------
        df["gene_id"] = df["attr"].str.extract(r'gene_id "([^"]+)"', expand=False)
        df["transcript_id"] = df["attr"].str.extract(
            r'transcript_id "([^"]+)"', expand=False
        )

        # -------- 3. exon 子集（再次 copy，后面要写列） --------
        df_exon = df.loc[df.feature == "exon"].copy()
        df_exon["attr"] = df_exon["attr"].fillna("")

        # -------- 4. exon_len --------
        exon_len = (
            df_exon.assign(exon_len=df_exon.end - df_exon.start + 1)
            .groupby(["gene_id", "transcript_id"], sort=False, observed=True)[
                "exon_len"
            ]
            .sum()
            .reset_index()
        )

        selected_tx: set[str] = set()

        # -------- 5. MANE_Select --------
        if prefer_mane:
            mane_tx = df_exon.loc[
                df_exon["attr"].str.contains("MANE_Select", na=False),
                "transcript_id",
            ].unique()
            selected_tx.update(mane_tx)

        # -------- 6. basic（排除已选） --------
        if prefer_basic:
            basic_tx = df_exon.loc[
                df_exon["attr"].str.contains('tag "basic"', na=False),
                "transcript_id",
            ].unique()
            selected_tx.update(set(basic_tx) - selected_tx)

        # -------- 7. 剩余基因 → 最长转录本 --------
        if selected_tx:
            used_genes = exon_len.loc[
                exon_len.transcript_id.isin(selected_tx), "gene_id"
            ]
            remaining = exon_len.loc[~exon_len.gene_id.isin(used_genes)]
        else:
            remaining = exon_len

        if not remaining.empty:
            idx = remaining.groupby("gene_id", sort=False)["exon_len"].idxmax()
            selected_tx.update(remaining.loc[idx, "transcript_id"])

        # -------- 8. 回写 annotation --------
        self.df_annotation = df.loc[df.transcript_id.isin(selected_tx)]

        return self

    def build_exon_index(self):
        """
        Build an exon-level transcript coordinate index.

        This method constructs a transcript-centric coordinate system
        by concatenating exons in transcriptional order and computing
        exon start offsets within each transcript.

        Strand handling:
            - '+' strand: exons ordered by increasing genomic start
            - '-' strand: exons ordered by decreasing genomic start

        Returns
        -------
        self.df_exon_index : pandas.DataFrame
            Exon-level index for fast genomic → transcript coordinate mapping, with columns:
                - gene_id
                - transcript_id
                - chrom
                - strand
                - start
                - end
                - exon_tx_offset
                - tx_len

        Notes
        -----
        This index is required for fast genomic → transcript coordinate mapping in ``map_A2G_to_meta``.
        """

        df = self.df_annotation
        df = df[df.feature == "exon"].copy()

        # exon length
        df["exon_len"] = df.end - df.start + 1

        # 排序：先 transcript，再按转录方向
        df.sort_values(
            ["transcript_id", "start"],
            inplace=True,
        )

        # 负链 exon 顺序反转
        neg = df.strand == "-"
        if neg.any():
            df_pos = df[df.strand == "+"].sort_values(
                ["transcript_id", "start"], ascending=[True, True]
            )
            df_neg = df[df.strand == "-"].sort_values(
                ["transcript_id", "start"], ascending=[True, False]
            )
            df = pd.concat([df_pos, df_neg], ignore_index=True)

        # exon 在 transcript 中的起始 offset
        df["exon_tx_offset"] = (
            df.groupby("transcript_id", sort=False)["exon_len"].cumsum()
            - df["exon_len"]
        )

        # transcript 总长度
        tx_len = (
            df.groupby("transcript_id", sort=False)["exon_len"].sum().rename("tx_len")
        )

        df = df.merge(tx_len, on="transcript_id", how="left")

        self.df_exon_index = df[
            [
                "gene_id",
                "transcript_id",
                "chrom",
                "strand",
                "start",
                "end",
                "exon_tx_offset",
                "tx_len",
            ]
        ]

        return self

    @default_to_self(("df_bases", "df_exon_index", "exclude_chrom"))
    def map_A2G_to_meta(
        self,
        df_bases: Optional[pd.DataFrame] = None,
        df_exon_index: Optional[pd.DataFrame] = None,
        exclude_chrom: Optional[Union[list, tuple]] = None,
        min_depth: int = 10,
    ):
        """
        Map per-base A→G statistics from genomic coordinates
        onto normalized transcript coordinates.

        This method intersects per-base A→G signals with exon intervals,
        converts genomic positions to transcript coordinates, and
        normalizes positions to the range [0, 1].

        Coverage Filtering
        ------------------
        Transcripts are retained if **at least one genomic base**
        within the transcript has sequencing depth ≥ ``min_depth``.

        Parameters
        ----------
        df_bases : pandas.DataFrame | None, optional
            Per-base A→G statistics table. If ``None``, uses ``self.df_bases``.

        df_exon_index : pandas.DataFrame | None, optional
            Exon-level transcript index. If ``None``, uses ``self.df_exon_index``.

        exclude_chrom : list | tuple | None, optional
            Chromosomes to exclude from analysis. If ``None``, uses ``self.exclude_chrom``.

        min_depth : int, optional
            Minimum per-base read depth required to retain a transcript, by default 10.

        Returns
        -------
        self.df_meta : pandas.DataFrame
            Per-base A→G signals mapped to normalized transcript coordinates.
            Columns: ["gene_id", "transcript_id", "tx_norm_pos", "alt_ratio", "depth"]

        Notes
        -----
        - Genomic coordinates are assumed to be 1-based and inclusive.
        - Internally, coordinates are converted to 0-based half-open
          intervals for PyRanges operations.
        - Strand information is respected when computing transcript
          positions.
        """

        bases = df_bases.copy()
        exon = df_exon_index.copy()

        if exclude_chrom is not None:
            bases = bases.loc[~bases.chrom.isin(exclude_chrom)]
            exon = exon.loc[~exon.chrom.isin(exclude_chrom)]

        # -------- bases → PyRanges --------
        bases = bases[
            ["chrom", "position", "strand", "alt_ratio", "chrom_site_reads_count"]
        ].rename(columns={"chrom_site_reads_count": "depth"})

        bases_df = pd.DataFrame(
            {
                "Chromosome": bases["chrom"].values,
                "Start": bases["position"].values - 1,  # 0-based
                "End": bases["position"].values,
                "Strand": bases["strand"].values,
                "alt_ratio": bases["alt_ratio"].values,
                "depth": bases["depth"].values,
            }
        )

        bases_pr = pr.PyRanges(bases_df)

        # -------- exon → PyRanges --------
        exon_df = pd.DataFrame(
            {
                "Chromosome": exon["chrom"].values,
                "Start": exon["start"].values - 1,
                "End": exon["end"].values,
                "Strand": exon["strand"].values,
                "gene_id": exon["gene_id"].values,
                "transcript_id": exon["transcript_id"].values,
                "exon_tx_offset": exon["exon_tx_offset"].values,
                "tx_len": exon["tx_len"].values,
            }
        )

        exon_pr = pr.PyRanges(exon_df)

        # -------- interval join --------
        joined = bases_pr.join(exon_pr)

        if joined.df.empty:
            self.df_meta = pd.DataFrame(
                columns=[
                    "gene_id",
                    "transcript_id",
                    "tx_norm_pos",
                    "alt_ratio",
                    "depth",
                ]
            )
            return self

        df = joined.df

        # -------- 筛除 so-called 总 reads < min_depth 的 transcript --------
        max_depth_per_tx = df.groupby("transcript_id")["depth"].max()
        df = df[df.transcript_id.map(max_depth_per_tx) >= min_depth]
        if df.empty:
            self.df_meta = pd.DataFrame(
                columns=[
                    "gene_id",
                    "transcript_id",
                    "tx_norm_pos",
                    "alt_ratio",
                    "depth",
                ]
            )
            return self

        # -------- transcript 坐标 --------
        fw = df["Strand"] == "+"

        df.loc[fw, "tx_pos"] = df.loc[fw, "exon_tx_offset"] + (
            df.loc[fw, "Start"] - df.loc[fw, "Start_b"]
        )

        df.loc[~fw, "tx_pos"] = df.loc[~fw, "exon_tx_offset"] + (
            df.loc[~fw, "End_b"] - df.loc[~fw, "Start"]
        )

        df["tx_norm_pos"] = df["tx_pos"] / df["tx_len"]

        self.df_meta = df[
            ["gene_id", "transcript_id", "tx_norm_pos", "alt_ratio", "depth"]
        ].copy()

        return self

    @default_to_self(("df_meta",))
    def bin_meta_A2G(
        self,
        df_meta: Optional[pd.DataFrame] = None,
        n_bins: int = 100,
        value: str = "alt_ratio",
        agg: str = "mean",
    ):
        """
        Bin meta-transcript A→G signals along normalized transcript length.

        Transcript-normalized positions in [0, 1] are discretized into
        equally sized bins, and values are aggregated per bin.

        Parameters
        ----------
        df_meta : pandas.DataFrame | None, optional
            Output of ``map_A2G_to_meta``. If ``None``, uses ``self.df_meta``.

        n_bins : int, optional
            Number of bins along the transcript, by default 100.

        value : str, optional
            Value to aggregate. One of:
            - "alt_ratio"
            - "depth"

        agg : str, optional
            Aggregation method:
            - "mean"
            - "sum"
            - "median"
            - "weighted" (depth-weighted mean of alt_ratio)

        Returns
        -------
        self.df_meta_plot : pandas.DataFrame
            Binned meta-transcript A→G profile for plotting.
            Columns: ["bin", ``value``]

        Raises
        ------
        ValueError
            If ``agg='weighted'`` is used with ``value!='alt_ratio'``.
        """

        df = df_meta.copy()
        df["bin"] = (df.tx_norm_pos * n_bins).astype(int)
        df["bin"] = df["bin"].clip(0, n_bins - 1)

        if agg == "weighted":
            if value != "alt_ratio":
                raise ValueError("agg='weighted' only valid when value='alt_ratio'")
            df["_weighted_alt"] = df["alt_ratio"] * df["depth"]

            grouped = df.groupby("bin", sort=True).agg(
                alt_sum=("_weighted_alt", "sum"),
                depth_sum=("depth", "sum"),
            )

            grouped["alt_ratio"] = grouped["alt_sum"] / grouped["depth_sum"]
            self.df_meta_plot = (
                grouped[["alt_ratio"]]
                .reset_index()
                .rename(columns={"alt_ratio": value})
            )
        else:
            self.df_meta_plot = (
                df.groupby("bin", sort=True)[value].agg(agg).reset_index()
            )

        return self

    @default_to_self(("df_meta_plot",))
    def plot_ax(
        self,
        df_meta_plot: Optional[pd.DataFrame] = None,
        n_bins: Optional[int] = None,
        ylabel: str = "A→G ratio",
        xlabel: str = "Normalized transcript position",
        title: Optional[str] = None,
        color: str = "#4485C7",
        lw: float = 2.0,
        label: Optional[str] = None,
        alpha: float = 0.8,
        ax: Optional[matplotlib.axes.Axes] = None,
        show: bool = True,
    ) -> matplotlib.axes.Axes:
        """
        Plot the binned meta-transcript A→G profile.

        This method visualizes the aggregated A→G signal along the
        normalized transcript (0 → 1).

        Parameters
        ----------
        df_meta_plot : pandas.DataFrame | None, optional
            Output of ``bin_meta_A2G``. If ``None``, uses ``self.df_meta_plot``.

        n_bins : int | None, optional
            Total number of bins. If ``None``, inferred from ``df_meta_plot``.

        ylabel : str, optional
            Y-axis label.

        xlabel : str, optional
            X-axis label.

        title : str | None, optional
            Figure title.

        color : str, optional
            Line color.

        lw : float, optional
            Line width.

        label : str | None, optional
            Line label for legend.

        alpha : float, optional
            Line transparency.

        ax : matplotlib.axes.Axes | None, optional
            Existing Axes to draw on. If ``None``, a new figure is created.

        show : bool, optional
            Whether to immediately call ``plt.show()``, by default True.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Matplotlib Axes containing the plotted meta profile.

        Notes
        -----
        - Missing bins are plotted as gaps (NaN).
        - Top and right spines are removed for a cleaner appearance.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        if n_bins is None:
            n_bins = df_meta_plot["bin"].max() + 1

        # ensure full x-axis
        x = np.arange(n_bins)
        y = np.full(n_bins, np.nan)
        y[df_meta_plot["bin"].values] = df_meta_plot.iloc[:, 1].values

        ax.plot(
            x / n_bins,
            y,
            color=color,
            lw=lw,
            label=label,
            alpha=alpha,
        )

        ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show:
            plt.tight_layout()
            plt.show()

        return ax
