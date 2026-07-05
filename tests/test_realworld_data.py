"""
Tests for the real-world dataset loaders (BibInteg and WDC-Product).

All tests use mock datasets only — no actual data download is required.
Tests verify:
  - Schema constants are self-consistent
  - FDs are respected by mock records
  - Config object is valid and accepted by data layer
  - Certification of canonical queries
  - Record → world conversion
  - load_dataset raises FileNotFoundError for missing data
  - CLI --info and --mock flags run without error
"""

from pathlib import Path

import pytest

import data.amazon_google as ag
import data.bibinteg as bib
import data.crosskg_dblp as ckg
import data.fodors_zagat as fz
import data.wdc as wdc
from data.utils import augmented_overlap

# ===========================================================================
# BibInteg
# ===========================================================================


class TestBibintegSchema:
    def test_n_attrs(self):
        assert bib.N_ATTRS == 7

    def test_view_attrs_subset(self):
        for v, attrs in bib.VIEW_SCHEMAS.items():
            assert attrs <= frozenset(range(bib.N_ATTRS))

    def test_overlaps_are_intersections(self):
        v0, v1, v2 = bib.VIEW_SCHEMAS[0], bib.VIEW_SCHEMAS[1], bib.VIEW_SCHEMAS[2]
        for overlap in bib.OVERLAP_SCHEMAS:
            # each overlap must be ⊆ at least two views
            in_views = sum(1 for v in [v0, v1, v2] if overlap <= v)
            assert in_views >= 2, f"Overlap {overlap} not shared by ≥2 views"

    def test_fd_lhs_in_attrs(self):
        for lhs, rhs in bib.FDS:
            assert lhs <= frozenset(range(bib.N_ATTRS))
            assert rhs < bib.N_ATTRS

    def test_config_valid(self):
        cfg = bib.CONFIG
        assert cfg.n_attrs == bib.N_ATTRS
        assert cfg.fds == bib.FDS


class TestBibintegMock:
    @pytest.fixture(scope="class")
    def records(self):
        return bib.make_mock_dataset(n=100, seed=0)

    def test_count(self, records):
        assert len(records) == 100

    def test_record_length(self, records):
        for r in records:
            assert len(r.to_world_row()) == bib.N_ATTRS

    def test_fds_respected(self, records):
        """All records with same (title,author,year) share venue,doi_prefix,n_authors."""
        seen: dict[tuple, tuple] = {}
        for r in records:
            key = (r[0], r[1], r[2])
            val = (r[3], r[4], r[5])
            if key in seen:
                assert seen[key] == val, f"FD violation at key={key}"
            else:
                seen[key] = val

    def test_decade_derived_from_year(self, records):
        for r in records:
            year = r[2]
            decade = r[6]
            assert decade == (year // 10) % 7

    def test_to_world_row_types(self, records):
        for r in records[:10]:
            row = r.to_world_row()
            assert all(isinstance(v, int) for v in row)


class TestBibintegCertification:
    def test_augmented_overlap(self):
        aug = augmented_overlap(bib.OVERLAP_SCHEMAS[0], bib.FDS)
        # closure of {0,1,2} under FDs should include 3,4,5 (and via {2}→6, attr 6 too)
        assert frozenset({0, 1, 2, 3, 4, 5}) <= aug

    def test_q_venue_certified(self):
        assert bib.Q_VENUE.is_certified(bib.CONFIG)

    def test_q_doi_certified(self):
        assert bib.Q_DOI.is_certified(bib.CONFIG)

    def test_q_large_team_certified(self):
        assert bib.Q_LARGE_TEAM.is_certified(bib.CONFIG)


class TestBibintegWorldConversion:
    def test_records_to_worlds_shape(self):
        records = bib.make_mock_dataset(n=10, seed=1)
        worlds = bib.records_to_worlds(records)
        assert len(worlds) == 10
        for w in worlds:
            assert len(w) == 1  # one tuple per world
            assert len(w[0]) == bib.N_ATTRS


class TestBibintegErrors:
    def test_load_raises_on_missing_data(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="dblp.csv"):
            bib.load_dataset(tmp_path)

    def test_download_raises_not_implemented(self):
        from data.bibinteg import _download

        with pytest.raises(NotImplementedError):
            _download(Path("/tmp/nonexistent"))


class TestBibintegCLI:
    def test_info_flag(self, capsys):
        bib.main(["--info"])
        out = capsys.readouterr().out
        assert "BibInteg schema" in out

    def test_mock_flag(self, capsys):
        bib.main(["--mock"])
        out = capsys.readouterr().out
        assert "Mock BibInteg" in out


# ===========================================================================
# WDC-Product
# ===========================================================================


class TestWdcSchema:
    def test_n_attrs(self):
        assert wdc.N_ATTRS == 8

    def test_view_attrs_subset(self):
        for v, attrs in wdc.VIEW_SCHEMAS.items():
            assert attrs <= frozenset(range(wdc.N_ATTRS))

    def test_overlaps_are_intersections(self):
        views = list(wdc.VIEW_SCHEMAS.values())
        for overlap in wdc.OVERLAP_SCHEMAS:
            in_views = sum(1 for v in views if overlap <= v)
            assert in_views >= 2, f"Overlap {overlap} not shared by ≥2 views"

    def test_fd_lhs_in_attrs(self):
        for lhs, rhs in wdc.FDS:
            assert lhs <= frozenset(range(wdc.N_ATTRS))
            assert rhs < wdc.N_ATTRS

    def test_config_valid(self):
        cfg = wdc.CONFIG
        assert cfg.n_attrs == wdc.N_ATTRS
        assert cfg.fds == wdc.FDS


class TestWdcMock:
    @pytest.fixture(scope="class")
    def records(self):
        return wdc.make_mock_dataset(n=99, seed=0)

    def test_count(self, records):
        # make_mock_dataset generates floor(n/3)*3 records (cluster-aligned)
        assert len(records) == 99

    def test_record_length(self, records):
        for r in records:
            assert len(r.to_world_row()) == wdc.N_ATTRS

    def test_fds_respected(self, records):
        """All records with same (brand,model) share category, price_bucket, in_stock."""
        seen: dict[tuple, tuple] = {}
        for r in records:
            key = (r[0], r[1])
            val = (r[2], r[3], r[7])  # category, price_bucket, in_stock
            if key in seen:
                assert seen[key] == val, f"FD violation at key={key}"
            else:
                seen[key] = val

    def test_attr_ranges(self, records):
        for r in records:
            row = r.to_world_row()
            assert 0 <= row[2] < 64  # category
            assert 0 <= row[3] < 16  # price_bucket
            assert 0 <= row[4] < 5  # rating_bucket
            assert 0 <= row[5] <= 10  # n_reviews_log
            assert 0 <= row[6] < 3  # source_id
            assert row[7] in (0, 1)  # in_stock

    def test_to_world_row_types(self, records):
        for r in records[:10]:
            row = r.to_world_row()
            assert all(isinstance(v, int) for v in row)


class TestWdcCertification:
    def test_augmented_overlap(self):
        aug = augmented_overlap(wdc.OVERLAP_SCHEMAS[0], wdc.FDS)
        # {0,1}→2,3,7; closure of {0,1,2} gets 3,7 too
        assert frozenset({0, 1, 2, 3, 7}) <= aug

    def test_q_available_certified(self):
        assert wdc.Q_AVAILABLE.is_certified(wdc.CONFIG)

    def test_q_cheap_certified(self):
        assert wdc.Q_CHEAP.is_certified(wdc.CONFIG)

    def test_q_highly_rated_not_certified(self):
        # attr 4 (rating_bucket) not in closure({0,1,2}) under FDS
        assert not wdc.Q_HIGHLY_RATED.is_certified(wdc.CONFIG)

    def test_q_reviewed_not_certified(self):
        # attr 5 (n_reviews_log) not in aug_overlap {0,1,2,3,7}
        assert not wdc.Q_REVIEWED.is_certified(wdc.CONFIG)

    def test_q_popular_not_certified(self):
        # footprint includes attrs 4 and 5, neither in aug_overlap
        assert not wdc.Q_POPULAR.is_certified(wdc.CONFIG)

    def test_clustered_mock_has_same_key_groups(self):
        """Clustered mock must have obs-key groups of size 3 (one per source)."""
        import numpy as np

        from data.utils import augmented_overlap
        from data.witness import observation_key

        aug = [augmented_overlap(o, wdc.FDS) for o in wdc.OVERLAP_SCHEMAS]
        records = wdc.make_mock_dataset(n=30, seed=7)  # 10 clusters × 3
        groups: dict = {}
        for r in records:
            w = np.array([r.to_world_row()], dtype=np.int32)
            key = observation_key(w, aug)
            groups[key] = groups.get(key, 0) + 1
        assert all(v == 3 for v in groups.values()), (
            "each cluster must produce 3 identical-key records"
        )


class TestWdcWorldConversion:
    def test_records_to_worlds_shape(self):
        records = wdc.make_mock_dataset(n=9, seed=2)  # 3 clusters × 3 sources = 9
        worlds = wdc.records_to_worlds(records)
        assert len(worlds) == 9
        for w in worlds:
            assert len(w) == 1
            assert len(w[0]) == wdc.N_ATTRS

    def test_records_to_cluster_worlds_count(self):
        records = wdc.make_mock_dataset(n=30, seed=3)  # 10 clusters × 3
        worlds = wdc.records_to_cluster_worlds(records)
        assert len(worlds) == 10

    def test_cluster_worlds_have_three_rows(self):
        records = wdc.make_mock_dataset(n=30, seed=3)
        worlds = wdc.records_to_cluster_worlds(records)
        for w in worlds:
            assert len(w) == 3
            for row in w:
                assert len(row) == wdc.N_ATTRS

    def test_cluster_world_rows_share_brand_model(self):
        records = wdc.make_mock_dataset(n=30, seed=4)
        worlds = wdc.records_to_cluster_worlds(records)
        for w in worlds:
            brand = w[0][0]
            model = w[0][1]
            for row in w:
                assert row[0] == brand, "all rows in cluster must share brand_hash"
                assert row[1] == model, "all rows in cluster must share model_hash"

    def test_cluster_world_source_ids_are_distinct(self):
        records = wdc.make_mock_dataset(n=30, seed=5)
        worlds = wdc.records_to_cluster_worlds(records)
        for w in worlds:
            src_ids = {row[6] for row in w}
            assert len(src_ids) == 3, "each cluster must have one record per source"

    def test_cluster_world_fds_hold(self):
        """FD-determined attributes must be equal within each cluster."""
        records = wdc.make_mock_dataset(n=30, seed=6)
        worlds = wdc.records_to_cluster_worlds(records)
        for w in worlds:
            category = w[0][2]
            price_bucket = w[0][3]
            in_stock = w[0][7]
            for row in w:
                assert row[2] == category
                assert row[3] == price_bucket
                assert row[7] == in_stock


class TestWdcErrors:
    def test_load_raises_on_missing_data(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="amazon.csv"):
            wdc.load_dataset(tmp_path)

    def test_download_raises_not_implemented(self):
        from data.wdc import _download

        with pytest.raises(NotImplementedError):
            _download(Path("/tmp/nonexistent"))


class TestWdcCLI:
    def test_info_flag(self, capsys):
        wdc.main(["--info"])
        out = capsys.readouterr().out
        assert "WDC-Product schema" in out

    def test_mock_flag(self, capsys):
        wdc.main(["--mock"])
        out = capsys.readouterr().out
        assert "Mock WDC-Product" in out


# ===========================================================================
# CrossKG-DBLP (OpenAlex × DBLP) — the real-witness dataset
# ===========================================================================


class TestCrossKGSchema:
    def test_n_attrs(self):
        assert ckg.N_ATTRS == 4

    def test_view_attrs_subset(self):
        for v, attrs in ckg.VIEW_SCHEMAS.items():
            assert attrs <= frozenset(range(ckg.N_ATTRS))

    def test_overlaps_are_intersections(self):
        views = list(ckg.VIEW_SCHEMAS.values())
        for overlap in ckg.OVERLAP_SCHEMAS:
            in_views = sum(1 for v in views if overlap <= v)
            assert in_views >= 2, f"Overlap {overlap} not shared by ≥2 views"

    def test_fd_lhs_in_attrs(self):
        for lhs, rhs in ckg.FDS:
            assert lhs <= frozenset(range(ckg.N_ATTRS))
            assert rhs < ckg.N_ATTRS

    def test_config_valid(self):
        cfg = ckg.CONFIG
        assert cfg.n_attrs == ckg.N_ATTRS
        assert cfg.fds == ckg.FDS


class TestCrossKGMock:
    @pytest.fixture(scope="class")
    def records(self):
        return ckg.make_mock_dataset(n=100, seed=0)

    def test_count(self, records):
        # two records (one per source) per paper
        assert len(records) == 200

    def test_record_length(self, records):
        for r in records:
            assert len(r.to_world_row()) == ckg.N_ATTRS

    def test_sources_paired(self, records):
        """Each paper contributes exactly one DBLP (src 0) and one OpenAlex (src 1) row."""
        by_doi: dict[int, list[int]] = {}
        for r in records:
            by_doi.setdefault(r[0], []).append(r[3])
        for doi_id, srcs in by_doi.items():
            assert sorted(srcs) == [0, 1], f"paper {doi_id} not paired across sources"

    def test_overlap_agrees_across_sources(self, records):
        """The two source-rows of a paper share doi_id and publisher_bit (the overlap)."""
        by_doi: dict[int, set] = {}
        for r in records:
            by_doi.setdefault(r[0], set()).add(r[1])  # publisher_bit
        for doi_id, pubs in by_doi.items():
            assert len(pubs) == 1, f"paper {doi_id} disagrees on publisher (overlap must agree)"

    def test_to_world_row_types(self, records):
        for r in records[:10]:
            assert all(isinstance(v, int) for v in r.to_world_row())


class TestCrossKGCertification:
    def test_augmented_overlap(self):
        aug = augmented_overlap(ckg.OVERLAP_SCHEMAS[0], ckg.FDS)
        assert aug == frozenset({0, 1})  # {0} closed under {0}->1

    def test_q_publisher_certified(self):
        # footprint {0,1} ⊆ Õ={0,1}; both sources share the DOI → no witness
        assert ckg.Q_PUBLISHER.is_certified(ckg.CONFIG)

    def test_q_large_team_not_certified(self):
        # attr 2 (large_team_bit) ∉ Õ; the two sources genuinely disagree
        assert not ckg.Q_LARGE_TEAM.is_certified(ckg.CONFIG)


class TestCrossKGWitness:
    def test_mock_contains_real_witness(self):
        """A straddling paper = two records, same obs key, different Q_large_team answer."""
        import numpy as np

        from data.witness import observation_key

        aug = [augmented_overlap(o, ckg.FDS) for o in ckg.OVERLAP_SCHEMAS]
        records = ckg.make_mock_dataset(n=200, seed=0)
        groups: dict[tuple, set] = {}
        for r in records:
            w = np.array([r.to_world_row()], dtype=np.int32)
            key = observation_key(w, aug)
            groups.setdefault(key, set()).add(r[2])  # large_team_bit
        witnesses = [k for k, bits in groups.items() if len(bits) > 1]
        assert witnesses, "mock dataset must contain at least one cross-source witness"


class TestCrossKGWorldConversion:
    def test_records_to_worlds_shape(self):
        records = ckg.make_mock_dataset(n=10, seed=1)
        worlds = ckg.records_to_worlds(records)
        assert len(worlds) == 20  # 10 papers × 2 sources
        for w in worlds:
            assert len(w) == 1
            assert len(w[0]) == ckg.N_ATTRS


class TestCrossKGErrors:
    def test_load_raises_on_missing_data(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="dblp.csv"):
            ckg.load_dataset(tmp_path)


class TestCrossKGCLI:
    def test_info_flag(self, capsys):
        ckg.main(["--info"])
        out = capsys.readouterr().out
        assert "CrossKG-DBLP" in out

    def test_mock_flag(self, capsys):
        ckg.main(["--mock"])
        out = capsys.readouterr().out
        assert "Mock CrossKG-DBLP" in out


# ===========================================================================
# Amazon-Google (product-domain real-witness dataset)
# ===========================================================================


class TestAmazonGoogleSchema:
    def test_n_attrs(self):
        assert ag.N_ATTRS == 4

    def test_overlaps_are_intersections(self):
        views = list(ag.VIEW_SCHEMAS.values())
        for overlap in ag.OVERLAP_SCHEMAS:
            assert sum(1 for v in views if overlap <= v) >= 2

    def test_fd_lhs_in_attrs(self):
        for lhs, rhs in ag.FDS:
            assert lhs <= frozenset(range(ag.N_ATTRS))
            assert rhs < ag.N_ATTRS

    def test_config_valid(self):
        assert ag.CONFIG.n_attrs == ag.N_ATTRS
        assert ag.CONFIG.fds == ag.FDS


class TestAmazonGoogleCertification:
    def test_augmented_overlap(self):
        assert augmented_overlap(ag.OVERLAP_SCHEMAS[0], ag.FDS) == frozenset({0, 1})

    def test_q_catalog_certified(self):
        assert ag.Q_CATALOG.is_certified(ag.CONFIG)

    def test_q_expensive_not_certified(self):
        assert not ag.Q_EXPENSIVE.is_certified(ag.CONFIG)


class TestAmazonGoogleMock:
    @pytest.fixture(scope="class")
    def records(self):
        return ag.make_mock_dataset(n=100, seed=0)

    def test_count(self, records):
        assert len(records) == 200  # one record per source per pair

    def test_record_length(self, records):
        for r in records:
            assert len(r.to_world_row()) == ag.N_ATTRS

    def test_overlap_agrees_across_sources(self, records):
        """The two source-rows of a pair share pair_id and catalog_bit (the overlap)."""
        by_pair: dict[int, set] = {}
        for r in records:
            by_pair.setdefault(r[0], set()).add(r[1])
        for pair_id, cats in by_pair.items():
            assert len(cats) == 1, f"pair {pair_id} disagrees on catalog_bit (overlap)"


class TestAmazonGoogleWitness:
    def test_mock_contains_real_witness(self):
        import numpy as np

        from data.witness import observation_key

        aug = [augmented_overlap(o, ag.FDS) for o in ag.OVERLAP_SCHEMAS]
        records = ag.make_mock_dataset(n=200, seed=0)
        groups: dict[tuple, set] = {}
        for r in records:
            w = np.array([r.to_world_row()], dtype=np.int32)
            groups.setdefault(observation_key(w, aug), set()).add(r[2])  # expensive_bit
        assert [k for k, bits in groups.items() if len(bits) > 1], "mock must contain a witness"


class TestAmazonGoogleWorldConversion:
    def test_records_to_worlds_shape(self):
        worlds = ag.records_to_worlds(ag.make_mock_dataset(n=10, seed=1))
        assert len(worlds) == 20
        for w in worlds:
            assert len(w) == 1 and len(w[0]) == ag.N_ATTRS


class TestAmazonGoogleErrors:
    def test_load_raises_on_missing_data(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="amazon.csv"):
            ag.load_dataset(tmp_path)


class TestAmazonGoogleCLI:
    def test_info_flag(self, capsys):
        ag.main(["--info"])
        assert "Amazon-Google" in capsys.readouterr().out

    def test_mock_flag(self, capsys):
        ag.main(["--mock"])
        assert "Mock Amazon-Google" in capsys.readouterr().out


# ===========================================================================
# Fodors-Zagat (restaurant-domain real-witness dataset)
# ===========================================================================


class TestFodorsZagatSchema:
    def test_n_attrs(self):
        assert fz.N_ATTRS == 4

    def test_overlaps_are_intersections(self):
        views = list(fz.VIEW_SCHEMAS.values())
        for overlap in fz.OVERLAP_SCHEMAS:
            assert sum(1 for v in views if overlap <= v) >= 2

    def test_fd_lhs_in_attrs(self):
        for lhs, rhs in fz.FDS:
            assert lhs <= frozenset(range(fz.N_ATTRS))
            assert rhs < fz.N_ATTRS

    def test_config_valid(self):
        assert fz.CONFIG.n_attrs == fz.N_ATTRS
        assert fz.CONFIG.fds == fz.FDS


class TestFodorsZagatCertification:
    def test_augmented_overlap(self):
        assert augmented_overlap(fz.OVERLAP_SCHEMAS[0], fz.FDS) == frozenset({0, 1})

    def test_q_segment_certified(self):
        assert fz.Q_SEGMENT.is_certified(fz.CONFIG)

    def test_q_cuisine_not_certified(self):
        assert not fz.Q_CUISINE.is_certified(fz.CONFIG)


class TestFodorsZagatMock:
    @pytest.fixture(scope="class")
    def records(self):
        return fz.make_mock_dataset(n=100, seed=0)

    def test_count(self, records):
        assert len(records) == 200

    def test_record_length(self, records):
        for r in records:
            assert len(r.to_world_row()) == fz.N_ATTRS

    def test_overlap_agrees_across_sources(self, records):
        """The two source-rows of a pair share pair_id and segment_bit (the overlap)."""
        by_pair: dict[int, set] = {}
        for r in records:
            by_pair.setdefault(r[0], set()).add(r[1])
        for pair_id, segs in by_pair.items():
            assert len(segs) == 1, f"pair {pair_id} disagrees on segment_bit (overlap)"


class TestFodorsZagatWitness:
    def test_mock_contains_real_witness(self):
        import numpy as np

        from data.witness import observation_key

        aug = [augmented_overlap(o, fz.FDS) for o in fz.OVERLAP_SCHEMAS]
        records = fz.make_mock_dataset(n=200, seed=0)
        groups: dict[tuple, set] = {}
        for r in records:
            w = np.array([r.to_world_row()], dtype=np.int32)
            groups.setdefault(observation_key(w, aug), set()).add(r[2])  # cuisine_bit
        assert [k for k, bits in groups.items() if len(bits) > 1], "mock must contain a witness"


class TestFodorsZagatErrors:
    def test_load_raises_on_missing_data(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="fodors.csv"):
            fz.load_dataset(tmp_path)


class TestFodorsZagatCLI:
    def test_info_flag(self, capsys):
        fz.main(["--info"])
        assert "Fodors-Zagat" in capsys.readouterr().out

    def test_mock_flag(self, capsys):
        fz.main(["--mock"])
        assert "Mock Fodors-Zagat" in capsys.readouterr().out
