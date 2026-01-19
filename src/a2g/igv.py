from .utils import default_to_self
import pysam
import numpy as np
import pandas as pd
import matplotlib.axes
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.ticker as ticker
import pyranges as pr
from typing import List, Tuple, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class IGV_a2g:
    """
    IGV-like visualization for A→G RNA editing signals on concatenated genomic intervals.

    This class provides an integrated plotting framework to visualize:
        - gene / transcript structure (concatenated blocks)
        - per-read alignments with A / A→G sites
        - per-base A→G ratio
        - per-base A-site read depth

    All genomic coordinates are first mapped onto a **continuous plot coordinate**
    defined by user-specified genomic intervals, enabling compact visualization of
    discontinuous regions (e.g. exons, PCR amplicons).

    Coordinate System
    -----------------
    - Genomic input coordinates:
        - 1-based, inclusive (as commonly used in BAM / TSV outputs)
    - Internal PyRanges coordinates:
        - 0-based, half-open
    - Plot coordinates:
        - 0-based, continuous across concatenated intervals

    Data Requirements
    -----------------
    1. Per-read A→G table (from ``extract_A2G_per_read.py``)
       Required columns:
        - read_id
        - chrom
        - strand
        - start            (1-based)
        - end              (1-based)
        - mapping_regions  (1-based)
        - A_positions      (semicolon-separated genomic positions)
        - A_to_G_positions (semicolon-separated genomic positions or empty)

    2. Per-base A→G table (from ``extract_A2G_per_base.py``)
       Required columns:
        - chrom
        - position       (1-based)
        - strand
        - total
        - ref_count
        - alt_count
        - ref_ratio
        - alt_ratio
        - chrom_site_reads_count

    Typical Workflow
    ----------------
    >>> igv = IGV_a2g(
    ...     chrom="chr1",
    ...     intervals=[(100, 200), (300, 350)],
    ...     strand="+",
    ...     df_A2G_per_read=df_reads,
    ...     df_A2G_per_base=df_bases,
    ...     df_annotation=df_anno,
    ... )
    >>> igv.build_gene_model() \\
    ...    .build_reads_track() \\
    ...    .build_bases_track()
    >>> fig = igv.plot_mod(
    ...     figsize=(10, 8),
    ...     show=["gene_body", "reads", "A_depth", "A2G_ratio"],
    ...     max_reads=50,
    ...     coverage_cutoff=0.9,
    ...     title="A→G editing profile",
    ...     show_intervals=True,
    ... )

    Built-in Track Types
    --------------------
    - ``gene_body`` :
        Concatenated genomic blocks with strand direction arrow
    - ``reads`` :
        Individual read alignments with optional A / A→G site markers
    - ``A_depth`` :
        Per-base A-site read depth (scaled bar track)
    - ``A2G_ratio`` :
        Per-base A→G ratio (bar track)

    Notes
    -----
    - Reads are plotted **from top to bottom**, ordered by coverage.
    - Track layout is computed globally to avoid overlaps between tracks.
    - This class does NOT perform alignment or variant calling; it assumes
      all inputs are precomputed tables.

    Attributes
    ----------
    chrom : str
        Chromosome or transcript name.
    intervals : list[tuple[int, int]]
        Genomic intervals to concatenate (1-based, inclusive).
    strand : str
        Strand to visualize ("+" or "-").
    coord_mapper : pyranges.PyRanges
        Mapping from genomic coordinates to plot coordinates.
    df_reads : pd.DataFrame
        Original per-read A→G statistics.
    df_bases : pd.DataFrame
        Original per-base A→G statistics.
    df_annotation : pd.DataFrame
        Preprocessed genomic annotation table for plotting gene body.
    gene_model : dict[str:list[tuple[int, int]]]
        Plot-ready gene structure data.
    df_read_plot : pd.DataFrame
        Plot-ready per-read data.
    df_base_plot : pd.DataFrame
        Plot-ready per-base data.
    gene_length : int
        Total length of concatenated plot coordinates.
    """

    def __init__(
        self,
        chrom: str,
        intervals: List[Tuple[int, int]],
        strand: str,
        df_A2G_per_read: pd.DataFrame,
        df_A2G_per_base: pd.DataFrame,
        df_annotation: pd.DataFrame = None,
    ):
        self.chrom = chrom
        self.intervals = intervals
        if strand in ("+", "-"):
            self.strand = strand
            self.df_reads = df_A2G_per_read
            self.df_bases = df_A2G_per_base
        else:
            raise ValueError("Not a valid strand symbol. (forward: '+', reverse: '-')")
        self.df_annotation = df_annotation
        self.coord_mapper = None
        self.gene_model = None
        self.df_read_plot = None
        self.df_base_plot = None
        self.gene_length = None

    @default_to_self(("df_annotation",))
    def select_repr_rnas(
        self,
        df_annotation: Optional[pd.DataFrame] = None,
    ):
        """
        Select one representative transcript per gene.

        For each gene, exactly one transcript is retained according to
        the following priority rules:

        Transcripts tagged as **"Ensembl_canonical"** and **"protein_coding"**.

        Parameters
        ----------
        df_annotation : pandas.DataFrame | None, optional
            GTF-like annotation table. If ``None``, uses ``self.df_annotation``.

        Returns
        -------
        self.df_annotation : pandas.DataFrame
            Containing only the selected representative transcripts.
        """

        df = df_annotation.loc[
            :, ["chrom", "start", "end", "strand", "feature", "attr"]
        ].copy()

        # ---- filters ----
        if self.exclude_chrom is not None:
            df = df.loc[~df.chrom.isin(self.exclude_chrom)]

        df = df.loc[
            df["attr"].str.contains('transcript_type "protein_coding"', na=False)
            & df["attr"].str.contains("Ensembl_canonical", na=False)
        ]

        # ---- parse ids ----
        df["gene_id"] = df["attr"].str.extract(r'gene_id "([^"]+)"', expand=False)
        df["transcript_id"] = df["attr"].str.extract(
            r'transcript_id "([^"]+)"', expand=False
        )
        self.df_annotation = df
        return self

    @default_to_self(("chrom", "intervals", "strand", "df_annotation"))
    def build_gene_model(
        self,
        chrom: Optional[str] = None,
        intervals: Optional[List[tuple[int, int]]] = None,
        strand: Optional[str] = None,
        df_annotation: Optional[pd.DataFrame] = None,
    ):
        """
        Build an IGV-like gene body model on concatenated plot coordinates.

        This method projects gene annotation features onto a continuous
        "plot coordinate" system constructed by concatenating multiple
        genomic intervals. The resulting gene model is intended for
        visualization (e.g. by :meth:`plot_gene_body`) rather than
        biological inference.

        The workflow consists of:

        1. Constructing a coordinate mapper that concatenates the provided
           genomic intervals into a single continuous axis.
        2. Filtering annotation records by strand.
        3. Projecting annotation features onto the concatenated coordinates
           using interval overlap.
        4. Collecting CDS, UTR, start_codon and stop_codon features directly.
        5. Inferring intron intervals *within each transcript* from exon
           ordering in plot coordinates.
        6. Deduplicating and sorting all feature intervals.

        Notes
        -----
        - Exons are **not** included in the final gene model.
          Introns are inferred from exon gaps within each transcript.
        - Coordinates in the gene model are in **plot coordinates**
          (0-based, half-open), not genomic coordinates.
        - If annotation does not overlap the concatenated intervals,
          the gene model is skipped with corresponding info.
        - This function does not raise on missing annotations; it logs
          informative messages instead.

        Coordinate conventions
        ----------------------
        - Input genomic intervals: 1-based, inclusive
        - Internal representation: 0-based, half-open
        - Plot coordinates: 0-based, continuous across all intervals

        Parameters
        ----------
        chrom : str, optional
            Chromosome name corresponding to the genomic intervals.
            If ``None``, defaults to ``self.chrom``.

        intervals : list[tuple[int, int]], optional
            Genomic intervals to concatenate, each given as
            ``(start, end)`` in 1-based inclusive coordinates.
            If ``None``, defaults to ``self.intervals``.

        strand : str, optional
            Strand used to filter annotation records.
            If ``None``, defaults to ``self.strand``.

        df_annotation : pandas.DataFrame, optional
            Pre-filtered annotation table. Must contain at least the
            following columns::

                chrom, start, end, strand, feature, transcript_id

            Coordinates are assumed to be 1-based and inclusive.

        Returns
        -------
        self.coord_mapper : pyranges.PyRanges
            Mapping from genomic coordinates to plot coordinates.
        self.gene_length : int
            Total length of concatenated plot coordinates.
        self.gene_model : dict[str, list[tuple[int, int]]]
            Gene body model with keys::

                "CDS", "UTR", "intron", "start_codon", "stop_codon"

            Each value is a list of (plot_start, plot_end) tuples.

        Raises
        ------
        ValueError
            If any interval is invalid (start < 1 or end < start).
        """

        # ---------- coordinate mapper ----------
        coord_mapper_record = []
        cursor = 0  # plot coordinate cursor

        for s, e in intervals:
            if s < 1 or e < s:
                raise ValueError(f"Invalid interval {(s, e)}")

            length = e - s + 1
            coord_mapper_record.append(
                {
                    "Chromosome": chrom,
                    "Start": s - 1,  # 0-based, half-open
                    "End": e,
                    "plot_start": cursor,
                    # "plot_end": cursor + length,
                }
            )
            cursor += length
        coord_mapper = pr.PyRanges(pd.DataFrame(coord_mapper_record))
        self.coord_mapper = coord_mapper
        self.gene_length = cursor

        # --------------------------------
        # ---------- gene model ----------
        if df_annotation is None:
            logger.info(
                "build_gene_model: df_annotation is None, skip gene body model."
            )
            return self

        # ---------- strand filter ----------
        df = df_annotation[df_annotation["strand"] == strand].copy()
        if df.empty:
            logger.info(
                "build_gene_model: no annotation left after strand filter (%s).", strand
            )
            return self

        # ---------- coordinate normalization ----------
        df["Start"] = df.start - 1
        df["End"] = df.end

        # ---------- build PyRanges ----------
        gr_anno = pr.PyRanges(
            df.rename(columns={"chrom": "Chromosome"})[
                [
                    "Chromosome",
                    "Start",
                    "End",
                    "feature",
                    "transcript_id",
                ]
            ]
        )

        # ---------- project to concat coordinates ----------
        ov = gr_anno.join(coord_mapper)
        df_ov = ov.df

        if df_ov.empty:
            logger.info(
                "build_gene_model: annotation does not overlap with concat intervals."
            )
            return self

        # ---------- compute plot coordinates ----------
        df_ov["ov_start"] = df_ov[["Start", "Start_b"]].max(axis=1)
        df_ov["ov_end"] = df_ov[["End", "End_b"]].min(axis=1)

        df_ov["plot_start"] = df_ov["plot_start"] + (
            df_ov["ov_start"] - df_ov["Start_b"]
        )
        df_ov["plot_end"] = df_ov["plot_start"] + (df_ov["ov_end"] - df_ov["ov_start"])

        # ---------- initialize gene body model ----------
        gene_model = {
            "CDS": [],
            "UTR": [],
            "intron": [],
            "start_codon": [],
            "stop_codon": [],
        }

        # ---------- collect CDS / UTR / codon ----------
        for r in df_ov.itertuples(index=False):
            if r.feature in gene_model and r.feature != "intron":
                gene_model[r.feature].append((r.plot_start, r.plot_end))

        # ---------- infer intron from exon (per transcript) ----------
        exon_df = df_ov[df_ov.feature == "exon"]

        for tid, sub in exon_df.groupby("transcript_id", sort=False):
            if len(sub) < 2:
                continue

            # concat 坐标下，plot_start 的大小顺序就是转录顺序
            sub = sub.sort_values("plot_start")

            exons = list(zip(sub.plot_start, sub.plot_end))
            for (s1, e1), (s2, e2) in zip(exons[:-1], exons[1:]):
                if e1 < s2:
                    gene_model["intron"].append((e1, s2))

        # ---------- sort & deduplicate ----------
        for k in gene_model:
            gene_model[k] = sorted(set(gene_model[k]))

        self.gene_model = gene_model
        return self

    @default_to_self(("df_reads", "strand", "coord_mapper"))
    def build_reads_track(
        self,
        df_reads: Optional[pd.DataFrame] = None,
        strand: Optional[str] = None,
        coord_mapper: Optional[pr.PyRanges] = None,
    ):
        """
        Convert per-read A→G statistics into plot-ready read segments.

        This method maps each read's genomic alignment onto the concatenated
        plot coordinate system and computes:
            - read segments in plot coordinates
            - A-site positions
            - A→G-site positions
            - read coverage over the gene model

        Only reads matching the specified strand are retained.

        Parameters
        ----------
        df_reads : pandas.DataFrame, optional
            Per-read A→G statistics table.
            If ``None``, defaults to ``self.df_reads``.

            Required columns:
                - read_id
                - chrom
                - strand
                - start              (1-based)
                - end                (1-based)
                - mapping_regions    (1-based)
                - A_positions        (semicolon-separated, 1-based)
                - A_to_G_positions   (semicolon-separated, 1-based or empty)

        strand : str, optional
            Strand to visualize ("+" or "-").
            If ``None``, defaults to ``self.strand``.

        coord_mapper : pyranges.PyRanges, optional
            Coordinate mapper produced by ``build_gene_model``.
            If ``None``, defaults to ``self.coord_mapper``.

        Returns
        -------
        self.df_read_plot : pandas.DataFrame
            Plot-ready per-read A→G table, with columns:
            - read_id
            - segments      : list of (plot_start, plot_end)
            - A_plot        : list of plot positions
            - A2G_plot      : list of plot positions
            - coverage      : fraction of gene model covered by the read

        Raises
        ------
        ValueError
            If no valid mapping regions are found after filtering reads by
            the specified strand. This usually indicates that the
            ``mapping_regions`` column is missing, empty, or all values
            are NaN for the selected strand.

        ValueError
            If no read segments remain after joining with ``coord_mapper``.
            This may be caused by strand mismatch, incompatible genomic
            intervals, or an incorrect coordinate mapper.

        Notes
        -----
        - Segment boundaries follow PyRanges semantics (0-based, half-open).
        - Coverage is computed as total aligned length divided by
          ``self.gene_length``.
        """

        df = df_reads[df_reads["strand"] == strand].copy()

        # ---------- 1. build read blocks ----------
        records = []
        for row in df.itertuples(index=False):
            if pd.isna(row.mapping_regions):
                continue
            for blk in row.mapping_regions.split(";"):
                s, e = map(int, blk.strip("()").split(","))
                records.append(
                    {
                        "Chromosome": row.chrom,
                        "Start": s - 1,  # 0-based
                        "End": e,  # half-open
                        "read_id": row.read_id,
                        "A_positions": row.A_positions,
                        "A2G_positions": row.A_to_G_positions,
                    }
                )

        if not records:
            raise ValueError(
                f"No valid mapping regions found in df_reads after filtering by strand '{strand}'. "
                f"Check that 'mapping_regions' column is present and not all values are NaN."
            )

        gr_reads = pr.PyRanges(pd.DataFrame(records))

        # ---------- 2. join with gene model ----------
        ov = gr_reads.join(coord_mapper)
        df_ov = ov.df
        if df_ov.empty:
            raise ValueError(
                f"No reads remain after joining with the gene model. "
                f"Check your strand filter and intervals. "
                f"Total input reads: {len(df)}."
            )

        # ---------- 3. compute overlap & plot segments ----------
        df_ov["ov_start"] = df_ov[["Start", "Start_b"]].max(axis=1)
        df_ov["ov_end"] = df_ov[["End", "End_b"]].min(axis=1)

        df_ov["seg_start"] = df_ov["plot_start"] + (
            df_ov["ov_start"] - df_ov["Start_b"]
        )
        df_ov["seg_end"] = df_ov["plot_start"] + (df_ov["ov_end"] - df_ov["Start_b"])

        # ---------- 4. aggregate per read ----------
        records = []
        for rid, sub in df_ov.groupby("read_id", sort=False):
            segments = (
                sub[["seg_start", "seg_end"]]
                .sort_values("seg_start")
                .itertuples(index=False, name=None)
            )
            segments = list(segments)
            if not segments:
                continue

            A_plot, A2G_plot = [], []

            for r in sub.itertuples(index=False):
                g_start = r.ov_start + 1
                g_end = r.ov_end
                plot_base = r.plot_start + (r.ov_start - r.Start_b)

                if pd.notna(r.A_positions):
                    A_pos = np.fromstring(r.A_positions, sep=";", dtype=int)
                    mask = (A_pos >= g_start) & (A_pos <= g_end)
                    A_plot.extend(plot_base + (A_pos[mask] - g_start))

                if pd.notna(r.A2G_positions):
                    A2G_pos = np.fromstring(r.A2G_positions, sep=";", dtype=int)
                    mask = (A2G_pos >= g_start) & (A2G_pos <= g_end)
                    A2G_plot.extend(plot_base + (A2G_pos[mask] - g_start))

            records.append(
                dict(
                    read_id=rid,
                    segments=segments,
                    A_plot=sorted(A_plot),
                    A2G_plot=sorted(A2G_plot),
                    five_prime=segments[0][0],
                    three_prime=segments[-1][1],
                    coverage=sum(e - s for s, e in segments) / self.gene_length,
                )
            )

        self.df_read_plot = pd.DataFrame(records)
        return self

    @default_to_self(("df_bases", "strand", "coord_mapper"))
    def build_bases_track(
        self,
        df_bases: Optional[pd.DataFrame] = None,
        coord_mapper: Optional[pr.PyRanges] = None,
        strand: Optional[str] = None,
    ):
        """
        Map per-base A→G statistics onto plot coordinates.

        This method converts genomic per-base A→G signals into the
        concatenated plot coordinate system using the gene model.

        Only bases on the specified strand are retained.

        Parameters
        ----------
        df_bases : pandas.DataFrame, optional
            Per-base A→G statistics table.
            If ``None``, defaults to ``self.df_bases``.

            Required columns:
                - chrom
                - position            (1-based)
                - strand
                - total               (read depth)
                - alt_ratio           (A→G ratio)

        coord_mapper : pyranges.PyRanges, optional
            Coordinate mapper produced by ``build_gene_model``.
            If ``None``, defaults to ``self.coord_mapper``.

        strand : str, optional
            Strand to visualize ("+" or "-").
            If ``None``, defaults to ``self.strand``.

        Returns
        -------
        self.df_base_plot : pandas.DataFrame
            Plot-ready per-base A→G table, with columns:
            - plot_pos   : Plot coordinate (0-based)
            - alt_ratio  : A→G ratio
            - depth      : Read depth at the site

        Notes
        -----
        Each genomic position is mapped independently; overlapping
        intervals are handled via PyRanges join semantics.
        """

        df = df_bases[df_bases["strand"] == strand].copy()
        df["Start"] = df.position - 1
        df["End"] = df.position

        gr_sites = pr.PyRanges(
            pd.DataFrame(
                {
                    "Chromosome": df.chrom,
                    "Start": df.Start,
                    "End": df.End,
                    "alt_ratio": df.alt_ratio,
                    "depth": df.total,
                }
            )
        )

        ov = gr_sites.join(coord_mapper)
        df_ov = ov.df.copy()
        df_ov["plot_pos"] = df_ov.plot_start + (df_ov.Start - df_ov.Start_b)

        self.df_base_plot = df_ov[["plot_pos", "alt_ratio", "depth"]]
        return self

    def plot_A2G_ratio(self, ax, df_base_plot, y=0.0, color="#DC923E"):
        """Plot per-base A→G ratio"""
        grid = [0.25 * 5, 0.5 * 5, 0.75 * 5, 1.0 * 5]
        for l in grid:
            ax.hlines(
                l + y,
                xmin=0,
                xmax=self.gene_length,
                color="grey",
                alpha=0.5,
                ls="--",
                lw=1.0,
                zorder=0.5,
            )
        ax.bar(
            df_base_plot.plot_pos,
            df_base_plot.alt_ratio * 5,
            width=0.9,
            bottom=y,
            color=color,
            alpha=0.9,
            align="edge",
        )
        ax.hlines(
            y,
            xmin=0,
            xmax=self.gene_length,
            lw=1.0,
            color="black",
            alpha=0.4,
            zorder=4,
            clip_on=False,
        )
        # ax.text(-5, y + 0.5, "A→G ratio", ha="right", va="center")
        return ax

    def plot_A_depth(self, ax, df_base_plot, y=1.0, color="#04C243"):
        """Plot per-base A depth"""
        ax.bar(
            df_base_plot.plot_pos,
            5,  # df_base_plot.depth,
            width=0.9,
            bottom=y,
            color=color,
            alpha=0.7,
            lw=1.0,
            align="edge",
        )
        ax.hlines(
            y,
            xmin=0,
            xmax=self.gene_length,
            lw=1.0,
            color="black",
            alpha=0.4,
            zorder=4,
            clip_on=False,
        )
        # ax.text(-5, y + 0.5, "A depth", ha="right", va="center")
        return ax

    def plot_gene_body(
        self,
        ax: matplotlib.axes.Axes,
        gene_model: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        *,
        y: float = 3.0,
        cds_height: float = 3.0,
        utr_height: float = 1.5,
        intron_height: float = 0.5,
        body_color: str = "black",
        utr_color: str = "black",
        intron_color: str = "black",
        show_codon: bool = False,
        codon_color: str = "yellow",
    ):
        """
        Plot an IGV-like gene body track on concatenated plot coordinates.

        This method visualizes a gene model composed of CDS, UTR, intron,
        start_codon and stop_codon features using a compact, IGV-style
        representation. Directional arrows are rendered directly on CDS
        and intron regions to indicate transcription direction.

        The function assumes all coordinates in ``gene_model`` are already
        mapped to the continuous plot coordinate system produced by
        :meth:`build_gene_model`.

        Rendering overview
        ------------------
        Features are drawn in the following order (from back to front layer):

        1. Introns
           Rendered as thinnest rectangles centered at ``y``.
        2. CDS
           Rendered as filled rectangles centered at ``y``.
        3. UTR
           Rendered as thinner rectangles centered at ``y``.
        4. Start / stop codons
           Rendered as vertical ticks centered on CDS height.
        5. Directional arrows
           Rendered on CDS and intron regions to indicate strand direction.

        Directional arrows
        ------------------
        - Arrows are placed only on CDS and intron regions.
        - Arrow positions are uniformly spaced across the gene body.
        - Arrow density adapts automatically to figure size to maintain
          approximately constant visual spacing in screen pixels.
        - If an arrow position overlaps both CDS and intron, CDS takes
          priority.
        - Arrow direction follows ``self.strand``.

        Arrow spacing is determined by:
        - The total gene length in plot coordinates.
        - The rendered axis width in pixels.
        - A minimum visual spacing (in pixels) between arrows.

        Coordinate conventions
        ----------------------
        - All coordinates are plot coordinates (0-based, half-open).
        - The x-axis corresponds to the concatenated genomic intervals.
        - The y-axis is in arbitrary plotting units.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes on which the gene body is drawn.

        gene_model : dict[str, list[tuple[int, int]]], optional
            Gene body model produced by :meth:`build_gene_model`.
            Expected keys include::

                "CDS", "UTR", "intron", "start_codon", "stop_codon"

            Each value is a list of (start, end) tuples in plot coordinates.
            If ``None`` or empty, the gene body track is skipped.

        y : float, optional
            Vertical center position of the gene body.
            Default is 3.0.

        cds_height : float, optional
            Height of CDS rectangles.
            Default is 3.0.

        utr_height : float, optional
            Height of UTR rectangles.
            Default is 1.5.

        intron_height : float, optional
            Height of intron rectangles.
            Default is 0.5.

        body_color : str, optional
            Fill color for CDS rectangles.

        utr_color : str, optional
            Fill color for UTR rectangles.

        intron_color : str, optional
            Fill color for intron rectangles.

        show_codon : bool, optional
            Whether to draw start and end codon markers.

        codon_color : str, optional
            Color used to draw start and stop codon markers.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The input axes, with the gene body track rendered.

        Notes
        -----
        - This function performs no coordinate validation.
        - It is intended purely for visualization.
        - Missing or empty gene models are silently skipped with logging.
        """

        if not gene_model:
            logger.info("No gene model, skip gene_body track.")
            return ax

        # 1. draw intron
        for s, e in gene_model.get("intron", []):
            ax.add_patch(
                patches.Rectangle(
                    (s, y - intron_height / 2),
                    e - s,
                    intron_height,
                    facecolor=intron_color,
                    edgecolor="none",
                    zorder=1,
                )
            )

        # 2. draw CDS
        for s, e in gene_model.get("CDS", []):
            ax.add_patch(
                patches.Rectangle(
                    (s, y - cds_height / 2),
                    e - s,
                    cds_height,
                    facecolor=body_color,
                    edgecolor="none",
                    zorder=2,
                )
            )

        # 3. draw UTR
        for s, e in gene_model.get("UTR", []):
            ax.add_patch(
                patches.Rectangle(
                    (s, y - utr_height / 2),
                    e - s,
                    utr_height,
                    facecolor=utr_color,
                    edgecolor="none",
                    zorder=2,
                )
            )

        # 4. draw start / stop codon
        if show_codon:
            for key in ("start_codon", "stop_codon"):
                for s, e in gene_model.get(key, []):
                    x = (s + e) / 2
                    ax.vlines(
                        x,
                        y - cds_height / 2,
                        y + cds_height / 2,
                        color=codon_color,
                        lw=1.5,
                        zorder=3,
                    )

        # 5. draw arrows
        direction = 1 if self.strand == "+" else -1

        # collect all CDS + intron intervals
        arrow_intervals = []
        for feature in ("CDS", "intron"):
            for s, e in gene_model.get(feature, []):
                arrow_intervals.append((s, e, feature))

        if not arrow_intervals:
            return

        gene_start = min(s for s, _, _ in arrow_intervals)
        gene_end = max(e for _, e, _ in arrow_intervals)
        gene_len = gene_end - gene_start

        # ---------- figure pixel width ----------
        ax_width_px = ax.get_window_extent().width
        min_spacing_px = 150  # 视觉上每个箭头至少 150px
        bp_per_pixel = gene_len / ax_width_px
        min_spacing_bp = min_spacing_px * bp_per_pixel

        # ---------- number of arrows ----------
        n_arrows = max(1, int(gene_len / min_spacing_bp))
        xs = np.linspace(gene_start, gene_end, n_arrows)

        # ---------- map arrows to CDS/intron ----------
        df_feat = pd.DataFrame(arrow_intervals, columns=["Start", "End", "feature"])
        df_feat["Chromosome"] = "plot"
        df_feat = pr.PyRanges(df_feat[["Chromosome", "Start", "End", "feature"]])

        gr_x = pr.PyRanges(
            pd.DataFrame({"Chromosome": "plot", "Start": xs, "End": xs + 1})
        )
        ov = gr_x.join(df_feat)
        if ov.df.empty:
            return

        df_arrow = ov.df[["Start", "feature"]].copy()
        df_arrow["priority"] = df_arrow["feature"].map({"CDS": 2, "intron": 1})
        df_arrow = df_arrow.sort_values("priority", ascending=False).drop_duplicates(
            "Start"
        )

        # ---------- draw ----------
        for r in df_arrow.itertuples(index=False):
            x_start = r.Start
            x_end = x_start + 0.1 * direction

            if r.feature == "CDS":
                fc = "white"
                ec = "white"
                z = 4
            else:  # intron
                fc = intron_color
                ec = intron_color
                z = 3

            ax.annotate(
                "",
                xy=(x_end, y),
                xytext=(x_start, y),
                arrowprops=dict(
                    arrowstyle="->",
                    edgecolor=ec,
                    facecolor=fc,
                    lw=2.0,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=25,  # 控制箭头头部大小
                ),
                zorder=z,
                clip_on=False,
            )

        return ax

    def plot_reads(
        self,
        ax: matplotlib.axes.Axes,
        df_read_plot: pd.DataFrame,
        y_start=2.0,
        max_reads=None,
        coverage_cutoff=0.9,
        show_A=True,
        show_A2G=True,
        read_color="#6da3d6",
        A_color="#228b22",
        A2G_color="#bf1f1f",
    ):
        """
        Plot individual read alignments with optional A and A→G markers.

        Reads are plotted from top to bottom, sorted by coverage.
        Each read is shown as one or more rectangular segments,
        with optional markers indicating A-sites and A→G sites.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to draw on.

        df_read_plot : pandas.DataFrame
            Output of ``build_reads_track``.

        y_start : float, optional
            Starting y-position (topmost read).

        max_reads : int or None, optional
            Maximum number of reads to draw.
            If ``None``, all reads passing the coverage filter are drawn.

        coverage_cutoff : float, optional
            Minimum coverage required for a read to be plotted,
            by default 0.9.

        show_A : bool, optional
            Whether to draw A-site markers, by default ``True``.

        show_A2G : bool, optional
            Whether to draw A→G-site markers, by default ``True``.

        read_color : str, optional
            Color of read segments.

        A_color : str, optional
            Color of A-site markers.

        A2G_color : str, optional
            Color of A→G-site markers.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The modified Axes object.
        """

        df_read_plot = df_read_plot.sort_values(
            by=["five_prime", "coverage"],
            ascending=[True, False],
            ignore_index=True,
        )
        if coverage_cutoff > 0:
            df_read_plot = df_read_plot[df_read_plot.coverage >= coverage_cutoff]
        if max_reads is not None:
            df_read_plot = df_read_plot.iloc[:max_reads]

        read_height = 0.3
        site_height = 0.3

        for i, row in enumerate(df_read_plot.itertuples(index=False)):
            y = y_start - i * 0.5
            y_mid = y + read_height / 2

            # ---- draw read backbone ----
            ax.hlines(
                y_mid,
                xmin=row.five_prime,
                xmax=row.three_prime,
                color="grey",
                lw=1.0,
                alpha=0.8,
                zorder=1.05,
            )

            # ---- draw read segments ----
            for s, e in row.segments:
                ax.add_patch(
                    patches.Rectangle(
                        (s, y),
                        e - s,
                        read_height,
                        color=read_color,
                        alpha=1.0,
                        zorder=1.1,
                        antialiased=False,
                    )
                )

            # ---- draw A sites ----
            if show_A and row.A_plot:
                for p in row.A_plot:
                    ax.add_patch(
                        patches.Rectangle(
                            (p, y + (read_height - site_height) / 2),
                            0.86,
                            site_height,
                            color=A_color,
                            alpha=1.0,
                            zorder=1.2,
                            antialiased=False,
                        )
                    )

            # ---- draw A→G sites ----
            if show_A2G and row.A2G_plot:
                for p in row.A2G_plot:
                    ax.add_patch(
                        patches.Rectangle(
                            (p, y + (read_height - site_height) / 2),
                            0.86,
                            site_height,
                            color=A2G_color,
                            alpha=1.0,
                            zorder=1.3,
                            antialiased=False,
                        )
                    )
        return ax

    @default_to_self(("coord_mapper", "gene_model", "df_read_plot", "df_base_plot"))
    def plot_mod(
        self,
        coord_mapper: Optional[pr.PyRanges] = None,
        gene_model: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        df_read_plot: Optional[pd.DataFrame] = None,
        df_base_plot: Optional[pd.DataFrame] = None,
        figsize: Optional[tuple] = (8, 10),
        title: Optional[str] = None,
        show: Optional[Union[list, tuple]] = (
            "gene_body",
            "reads",
            "A_depth",
            "A2G_ratio",
        ),
        track_gap: Optional[float] = 3.0,
        show_intervals: bool = True,
        **kwargs,
    ):
        """
        Assemble multiple visualization tracks into a single IGV-like figure.

        This method is the main high-level plotting interface of ``IGV_a2g``.
        It computes a global vertical layout and renders multiple visualization
        tracks on a shared x-axis using concatenated plot coordinates.

        Tracks are drawn from top to bottom in the order specified by ``show``.
        Each track occupies a dynamically computed vertical region to avoid
        overlap while preserving an IGV-like appearance.

        Supported track types
        ---------------------
        - ``"gene_body"`` : Gene structure (CDS / UTR / intron / codons)
        - ``"reads"``     : Per-read alignment blocks
        - ``"A_depth"``   : Per-base A coverage depth
        - ``"A2G_ratio"`` : Per-base A→G editing ratio

        Coordinate conventions
        ----------------------
        - All tracks share the same x-axis in plot coordinates.
        - Plot coordinates are continuous and produced by concatenating
          genomic intervals via ``coord_mapper``.

        Parameters
        ----------
        coord_mapper : pyranges.PyRanges, optional
            Coordinate mapper defining the concatenated plot coordinates.
            Defaults to ``self.coord_mapper``.

        gene_model : dict[str, list[tuple[int, int]]], optional
            Gene body model used by the ``gene_body`` track.
            Defaults to ``self.gene_model``.

        df_read_plot : pandas.DataFrame, optional
            Plot-ready per-read table used by the ``reads`` track.
            Defaults to ``self.df_read_plot``.

        df_base_plot : pandas.DataFrame, optional
            Plot-ready per-base table used by ``A_depth`` and ``A2G_ratio`` tracks.
            Defaults to ``self.df_base_plot``.

        figsize : tuple, optional
            Figure size given as ``(width, height)`` in inches.
            Default is ``(8, 10)``.

        title : str or None, optional
            Figure title.
            Default is ``None``.

        show : iterable of str, optional
            Track types to display, in drawing order from top to bottom.
            Default is ``("gene_body", "reads", "A_depth", "A2G_ratio")``.

        track_gap : float, optional
            Vertical spacing between adjacent tracks.
            Default is 3.0.

        show_intervals : bool, optional
            Whether to draw vertical guide lines at concatenated interval boundaries.
            Default is ``True``.

        **kwargs
        --------
        Additional keyword arguments forwarded to track-specific plotting functions.

        Reads track options

        - ``max_reads`` : Maximum number of reads to display. Default is ``None`` (draw all reads).
        - ``coverage_cutoff`` : Minimum read coverage required for a read to be included. Default is ``0.9``.
        - ``show_A`` : Whether to highlight A bases in reads. Default is ``True``.
        - ``show_A2G`` : Whether to highlight A→G editing events in reads. Default is ``True``.


        Gene body track options

        - ``body_color`` : Fill color of CDS regions. Default is ``"black"``.
        - ``utr_color`` : Fill color of UTR regions. Default is ``"black"``.
        - ``intron_color`` : Color used to draw intron lines and intron arrows. Default is ``"black"``.
        - ``show_codon`` : Whether to draw start and stop codon markers. Default is ``False``.
        - ``codon_color`` : Color used to draw start and stop codon markers. Default is ``"yellow"``.


        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object containing all requested tracks.

        Raises
        ------
        ValueError
            If an unknown track type is specified in ``show``.

        Notes
        -----
        - Track heights are computed dynamically.
        - The height of the ``reads`` track scales with the number of reads
          being drawn.
        - This method does not perform coordinate validation and assumes
          all inputs are already mapped to plot coordinates.
        """

        fig, ax = plt.subplots(figsize=figsize)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.xaxis.set_ticks_position("none")

        # ---------- track config ----------
        df_reads_filt = df_read_plot[
            df_read_plot.coverage >= kwargs.get("coverage_cutoff", 0.9)
        ]

        if kwargs.get("max_reads") is not None:
            n_drawn_reads = min(len(df_reads_filt), kwargs["max_reads"])
        else:
            n_drawn_reads = len(df_reads_filt)

        track_height = {
            "A2G_ratio": 5,
            "A_depth": 5,
            "gene_body": 3,
            "reads": n_drawn_reads * 0.5,
        }

        # ---------- layout (from top to bottom) ----------
        total_height = sum(track_height.get(name, 0) for name in show) + track_gap * (
            len(show) - 1
        )

        cursor = total_height
        track_y = {}

        for name in show:
            h = track_height.get(name, 0)

            if name == "reads":
                # reads 使用“顶部 y”
                track_y[name] = cursor
            else:
                # 其它 track 使用“底部 y”
                track_y[name] = cursor - h

            cursor -= h + track_gap

        # ---------- draw ----------
        for name in show:
            y = track_y[name]

            if name == "gene_body":
                self.plot_gene_body(
                    ax,
                    y=y,
                    gene_model=gene_model,
                    body_color=kwargs.get("body_color", "black"),
                    utr_color=kwargs.get("utr_color", "black"),
                    intron_color=kwargs.get("intron_color", "black"),
                    show_codon=kwargs.get("show_codon", False),
                    codon_color=kwargs.get("codon_color", "yellow"),
                )

            elif name == "A2G_ratio":
                self.plot_A2G_ratio(
                    ax,
                    df_base_plot,
                    y=y,
                    color="#DC923E",
                )

            elif name == "A_depth":
                self.plot_A_depth(
                    ax,
                    df_base_plot,
                    y=y,
                    color="#228B22",
                )

            elif name == "reads":
                self.plot_reads(
                    ax,
                    df_read_plot,
                    y_start=y,
                    max_reads=kwargs.get("max_reads"),
                    coverage_cutoff=kwargs.get("coverage_cutoff", 0.9),
                    show_A=kwargs.get("show_A", True),
                    show_A2G=kwargs.get("show_A2G", True),
                )

            else:
                raise ValueError(f"Unknown track: {name}")

        # ---------- draw genomic interval lines ----------
        if show_intervals and coord_mapper is not None and not coord_mapper.df.empty:
            interval_positions = sorted(set(coord_mapper.df["plot_start"].tolist()))[1:]

            for pos in interval_positions:
                ax.axvline(
                    pos,
                    color="grey",
                    linestyle="-",
                    lw=0.8,
                    alpha=0.3,
                    zorder=10,
                )

        # ---------- axis ----------
        ax.set_yticks([])
        ax.set_xlim(0, self.gene_length)
        ax.set_title(title)

        plt.tight_layout()
        return fig
