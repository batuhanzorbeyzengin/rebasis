"""Profile flags for models rebasis has never seen.

The table covers fourteen models. Everything else arrives as flags, and until
these existed the error told users to pass options that no command accepted —
which meant an unregistered model could not be measured at all.

The rule these tests defend is the one that matters: **an unset flag must not
overwrite a known value.** Turning "I did not say" into "there is no prefix" is
precisely the silent misconfiguration these flags exist to prevent.
"""

from __future__ import annotations

import pytest

from rebasis.cli._profiles import ProfileOverrides, resolve_profile
from rebasis.errors import ConfigError, UnknownModelProfile

pytestmark = pytest.mark.unit

KNOWN = "BAAI/bge-small-en-v1.5"
UNKNOWN = "someone/a-model-that-does-not-exist"


class TestUnregisteredModels:
    def test_a_dimension_is_enough(self) -> None:
        profile = resolve_profile(UNKNOWN, ProfileOverrides(dim=512))

        assert profile.dim == 512
        assert profile.model_id == UNKNOWN

    def test_without_a_dimension_it_still_refuses(self) -> None:
        """rebasis will not invent a dimension any more than a prefix."""
        with pytest.raises(UnknownModelProfile):
            resolve_profile(UNKNOWN)

    def test_prefixes_are_carried_through(self) -> None:
        profile = resolve_profile(
            UNKNOWN,
            ProfileOverrides(dim=768, query_prefix="query: ", document_prefix="passage: "),
        )

        assert profile.query_prefix == "query: "
        assert profile.document_prefix == "passage: "
        assert not profile.symmetric

    def test_a_prefix_without_a_dimension_says_which_is_missing(self) -> None:
        """Failing inside `profile_for` would report the missing dimension
        without mentioning that the prefix was accepted, which reads as though
        the prefix were the problem."""
        with pytest.raises(ConfigError, match="not its dimension"):
            resolve_profile(UNKNOWN, ProfileOverrides(query_prefix="query: "))

    def test_the_index_dimension_serves_as_a_fallback(self) -> None:
        """The index is authoritative about the model it was built with, so an
        unregistered *old* model needs no flag."""
        profile = resolve_profile(UNKNOWN, ProfileOverrides(), fallback_dim=384)

        assert profile.dim == 384

    def test_an_explicit_flag_outranks_the_fallback(self) -> None:
        profile = resolve_profile(UNKNOWN, ProfileOverrides(dim=1024), fallback_dim=384)

        assert profile.dim == 1024


class TestKnownModels:
    def test_the_table_is_used_when_nothing_is_overridden(self) -> None:
        profile = resolve_profile(KNOWN)

        assert profile.dim == 384
        assert profile.query_prefix

    def test_an_unset_flag_does_not_erase_a_known_prefix(self) -> None:
        """The rule this module exists for.

        Passing `--new-dim` while saying nothing about prefixes must not silently
        turn an asymmetric model into a symmetric one.
        """
        profile = resolve_profile(KNOWN, ProfileOverrides(dim=384))

        assert profile.query_prefix
        assert not profile.symmetric

    def test_a_flag_overrides_the_table(self) -> None:
        """Model cards change, and a user who knows better must be able to say so."""
        profile = resolve_profile(KNOWN, ProfileOverrides(query_prefix="custom: "))

        assert profile.query_prefix == "custom: "

    def test_the_fingerprint_changes_when_a_prefix_does(self) -> None:
        """The fingerprint is what makes a wrong adapter unloadable. If an
        overridden prefix did not change it, an adapter fitted with one prefix
        would load happily against an index built with another."""
        default = resolve_profile(KNOWN)
        overridden = resolve_profile(KNOWN, ProfileOverrides(query_prefix="custom: "))

        assert default.fingerprint() != overridden.fingerprint()


class TestEmptyOverrides:
    def test_no_flags_is_falsey(self) -> None:
        assert not ProfileOverrides()

    def test_any_flag_is_truthy(self) -> None:
        assert ProfileOverrides(dim=384)
        assert ProfileOverrides(query_prefix="q: ")

    def test_unset_fields_are_omitted_rather_than_passed_as_none(self) -> None:
        assert ProfileOverrides(dim=384).as_kwargs() == {}
