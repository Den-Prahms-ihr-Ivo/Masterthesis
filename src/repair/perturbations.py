"""Apply and sample the perturbation events from exposé 4.4.

apply_event returns a new instance using with_volunteer() or with_shift().
The input stays unchanged, so earlier steps can still be inspected.

Events are sampled from the evolving instance and S0, without seeing a repair
arm's current schedule. Otherwise a policy could change the events that it and
the other policies face. This is DESIGN.md rule 5; the test
test_no_sampler_accepts_an_arm_schedule checks the sampler signatures.

Rules (k = perturbation scale, instance.meta.scale unless given):
P1  +k or -k with probability 1/2 each, on the current d_s (after earlier P1 events).
    increase targets: shifts whose staffing in S0 <= current d_s;
      fallback: any shift open in S0 (fallback=True).
    decrease targets: shifts with current d_s - k >= 1;
      fallback: the event becomes an increase (fallback=True).
    Independent targeting (s0=None): increase targets = every shift.
P2  k distinct (w, p) pairs: w currently available in p AND assigned to a shift in
    period p in S0. Independent: any currently available pair.
P3  k distinct (w, r) pairs: w assigned to a shift of role r in S0; p^role_wr is
    redrawn uniformly from its two other values. Independent: any (w, r) pair.
13.9 (decided 2026-11-13): if P2 / P3 has fewer than k eligible targets, take all of
    them and top up from the independent set (fallback=True); ValueError if even that
    set is too small.

RQ1 profiles (P1, P2, P3 counts): balanced 5/5/5, availability 3/9/3, preference 3/3/9,
    in random order, sampled step by step against the evolving instance.
RQ2: step 1 removes one S0-assigned pair (the selected pair); step 6 restores it; the
    other 13 steps carry 4 P1, 5 P2, 4 P3, drawn independently of S0 and never
    touching the selected pair (P2 excludes it, P3 excludes its volunteer).

Randomness: np.random.SeedSequence(seed).spawn(len(STREAMS)), one Generator per name,
so e.g. the RQ2 background does not depend on which pair was selected from S0.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from typing import TypeVar

import numpy as np

from repair.model import (
    AvailabilityRemoval,
    AvailabilityRestoration,
    DemandChange,
    Event,
    Instance,
    PreferenceChange,
    Schedule,
    Sequence,
    SequenceKind,
    SequenceProfile,
)

# A target: a shift id (P1) or a (volunteer, period / role) pair (P2, P3)
T = TypeVar("T", int, tuple[int, int])

# (P1, P2, P3) counts per RQ1 profile; each sums to the horizon T = 15
PROFILES: dict[SequenceProfile, tuple[int, int, int]] = {
    SequenceProfile.BALANCED: (5, 5, 5),
    SequenceProfile.AVAILABILITY: (3, 9, 3),
    SequenceProfile.PREFERENCE: (3, 3, 9),
}
RQ2_BACKGROUND: tuple[int, int, int] = (4, 5, 4)  # P1, P2, P3 on the other 13 steps
RQ2_FORCING_STEP = 1  # 1-based: events[0]
RQ2_RESTORATION_STEP = 6  # 1-based: events[5]

TARGETING = ("s0", "independent")

# One sub-stream per purpose (see the module docstring)
STREAMS = ("order", "selected", "p1", "p2", "p3")


def _make_streams(seed: int) -> dict[str, np.random.Generator]:
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))

    return {
        name: np.random.default_rng(child)
        for name, child in zip(STREAMS, children, strict=True)
    }


def _event_order(
    counts: tuple[int, int, int], rng: np.random.Generator
) -> tuple[int, ...]:
    """1 = P1, 2 = P2, 3 = P3, each repeated by its count, in random order."""
    labels = [
        event_type
        for event_type, count in enumerate(counts, start=1)
        for _ in range(count)
    ]

    return tuple(int(value) for value in rng.permutation(labels))


def _sample_event(
    event_type: int,
    instance: Instance,
    s0: Schedule | None,
    k: int,
    streams: dict[str, np.random.Generator],
    selected: tuple[int, int] | None = None,
) -> Event:
    """One event of the given type; `selected` (RQ2) is excluded from P2 and P3."""
    if event_type == 1:
        return sample_p1(instance, s0, k, streams["p1"])

    if event_type == 2:
        return sample_p2(
            instance,
            s0,
            k,
            streams["p2"],
            exclude=selected,
        )

    if event_type == 3:
        return sample_p3(
            instance,
            s0,
            k,
            streams["p3"],
            exclude_volunteer=None if selected is None else selected[0],
        )

    raise ValueError(f"unknown event type: {event_type}")


def _check_index(index: int, size: int, label: str) -> None:
    if type(index) is not int or not 0 <= index < size:
        raise ValueError(f"invalid {label}: {index!r}")


def _check_scale(k: int) -> None:
    if type(k) is not int or k <= 0:
        raise ValueError("perturbation scale must be a positive integer")


def _choose(
    candidates: Iterable[T], count: int, rng: np.random.Generator
) -> tuple[T, ...]:
    """Uniform subset, returned in sorted order."""
    pool = sorted(candidates)  # sorted: draws must not depend on set order

    if count == 0:
        return ()

    indices = rng.choice(len(pool), size=count, replace=False)

    return tuple(pool[int(i)] for i in sorted(indices))


def _targets(
    eligible: Iterable[T], independent: Iterable[T], k: int, rng: np.random.Generator
) -> tuple[tuple[T, ...], bool]:
    """Return (targets, fallback), with eligible targets before top-ups."""
    _check_scale(k)

    eligible = set(eligible)
    independent = set(independent)

    if len(independent) < k:
        raise ValueError(f"need {k} targets, but only {len(independent)} are available")

    if len(eligible) >= k:
        return _choose(eligible, k, rng), False

    topups = _choose(
        independent - eligible,
        k - len(eligible),
        rng,
    )

    return tuple(sorted(eligible)) + topups, True


def apply_event(instance: Instance, event: Event) -> Instance:
    """Apply one event and return the changed instance.

    Reject invalid targets, invalid values and events that change no input data.
    This differs from an event being inert for a policy: a valid event can still
    leave that policy's preferred schedule unchanged.

    DemandChange(shift, delta): d_s += delta; ValueError if the result is < 1.
    AvailabilityRemoval(pairs): remove each period from that volunteer's available set;
        ValueError if a period is not available.
    PreferenceChange(changes): p^role_wr := new; ValueError if new equals the current
        value or is outside 0..2.
    AvailabilityRestoration(volunteer, period): add the period back; ValueError if it is
        already available.
    Everything else stays the same.
    """
    if isinstance(event, DemandChange):
        _check_index(event.shift, len(instance.shifts), "shift")

        if type(event.delta) is not int or event.delta == 0:
            raise ValueError("demand delta must be a nonzero integer")

        shift = instance.shifts[event.shift]
        demand = shift.demand + event.delta

        if demand < 1:
            raise ValueError("demand change would reduce demand below 1")

        return instance.with_shift(replace(shift, demand=demand))

    if isinstance(event, AvailabilityRemoval):
        if not event.pairs:
            raise ValueError("availability removal has no targets")

        if len(set(event.pairs)) != len(event.pairs):
            raise ValueError("duplicate availability-removal target")

        current = instance

        for w, p in event.pairs:
            _check_index(w, len(current.volunteers), "volunteer")
            _check_index(p, current.n_periods, "period")

            volunteer = current.volunteers[w]

            if p not in volunteer.available:
                raise ValueError(f"volunteer {w} is unavailable in period {p}")

            current = current.with_volunteer(
                replace(
                    volunteer,
                    available=volunteer.available - {p},
                )
            )

        return current

    if isinstance(event, PreferenceChange):
        if not event.changes:
            raise ValueError("preference change has no targets")

        targets = [(w, r) for w, r, _ in event.changes]
        if len(set(targets)) != len(targets):
            raise ValueError("duplicate preference-change target")

        current = instance

        for w, r, new in event.changes:
            _check_index(w, len(current.volunteers), "volunteer")
            _check_index(r, current.n_roles, "role")

            if type(new) is not int or new not in (0, 1, 2):
                raise ValueError(f"invalid preference value: {new!r}")

            volunteer = current.volunteers[w]

            if new == volunteer.pref_role[r]:
                raise ValueError(f"preference ({w}, {r}) would not change")

            preferences = list(volunteer.pref_role)
            preferences[r] = new

            current = current.with_volunteer(
                replace(volunteer, pref_role=tuple(preferences))
            )

        return current

    if isinstance(event, AvailabilityRestoration):
        _check_index(event.volunteer, len(instance.volunteers), "volunteer")
        _check_index(event.period, instance.n_periods, "period")

        volunteer = instance.volunteers[event.volunteer]

        if event.period in volunteer.available:
            raise ValueError("restored period is already available")

        return instance.with_volunteer(
            replace(
                volunteer,
                available=volunteer.available | {event.period},
            )
        )

    raise ValueError(f"unsupported event type: {type(event).__name__}")


def instances_along(instance: Instance, events: tuple[Event, ...]) -> list[Instance]:
    """[instance_1, ..., instance_T]: instance_t has events 1..t applied (a prefix fold).

    The base instance itself is not in the list; no events -> [].
    """
    result = []
    current = instance

    for event in events:
        current = apply_event(current, event)
        result.append(current)

    return result


def sample_p1(
    instance: Instance, s0: Schedule | None, k: int, rng: np.random.Generator
) -> DemandChange:
    """One P1 event on the current instance; s0=None -> independent targeting.

    Direction first (+k / -k with probability 1/2), then a uniform target from the
    eligible shifts of that direction (rules in the module docstring). ValueError if no
    shift qualifies even after the fallbacks.
    """
    _check_scale(k)

    delta = k if int(rng.integers(2)) == 0 else -k
    fallback = False
    candidates = []

    if delta < 0:
        candidates = [shift.id for shift in instance.shifts if shift.demand - k >= 1]

        if not candidates:
            delta = k
            fallback = True

    if delta > 0:
        if s0 is None:
            candidates = [shift.id for shift in instance.shifts]
        else:
            staffing = Counter(s for _, s in s0)

            candidates = [
                shift.id
                for shift in instance.shifts
                if staffing[shift.id] <= shift.demand
            ]

            if not candidates:
                candidates = [
                    shift.id for shift in instance.shifts if staffing[shift.id] > 0
                ]
                fallback = True

    if not candidates:
        raise ValueError("P1 has no target after fallbacks")

    shift_id = _choose(candidates, 1, rng)[0]

    return DemandChange(
        shift=shift_id,
        delta=delta,
        fallback=fallback,
    )


def sample_p2(
    instance: Instance,
    s0: Schedule | None,
    k: int,
    rng: np.random.Generator,
    exclude: tuple[int, int] | None = None,
) -> AvailabilityRemoval:
    """k distinct (w, p) pairs to make unavailable; s0=None -> independent targeting.

    `exclude` (RQ2: the selected pair) is never chosen. Fewer than k eligible -> top up
    (13.9), fallback=True. Pairs in a fixed order: eligible ones first, then top-ups.
    """
    independent = {
        (volunteer.id, period)
        for volunteer in instance.volunteers
        for period in volunteer.available
        if (volunteer.id, period) != exclude
    }

    if s0 is None:
        eligible = independent
    else:
        assigned = {(w, instance.shifts[s].period) for w, s in s0}
        eligible = assigned & independent

    pairs, fallback = _targets(eligible, independent, k, rng)

    return AvailabilityRemoval(
        pairs=pairs,
        fallback=fallback,
    )


def sample_p3(
    instance: Instance,
    s0: Schedule | None,
    k: int,
    rng: np.random.Generator,
    exclude_volunteer: int | None = None,
) -> PreferenceChange:
    """k distinct (w, r) pairs, each with a new p^role_wr != the current one;
    s0=None -> independent targeting.

    `exclude_volunteer` (RQ2: the selected volunteer) is never chosen. Fewer than k
    eligible -> top up (13.9), fallback=True.
    """
    independent = {
        (volunteer.id, role)
        for volunteer in instance.volunteers
        if volunteer.id != exclude_volunteer
        for role in range(instance.n_roles)
    }

    if s0 is None:
        eligible = independent
    else:
        assigned = {(w, instance.shifts[s].role) for w, s in s0}
        eligible = assigned & independent

    targets, fallback = _targets(eligible, independent, k, rng)

    changes = []

    for w, role in targets:
        old = instance.volunteers[w].pref_role[role]
        alternatives = [value for value in (0, 1, 2) if value != old]
        new = int(rng.choice(alternatives))

        changes.append((w, role, new))

    return PreferenceChange(
        changes=tuple(changes),
        fallback=fallback,
    )


def rq1_sequence(
    instance: Instance,
    s0: Schedule,
    profile: SequenceProfile,
    seed: int,
    horizon: int = 15,
    targeting: str = "s0",
    scale: int | None = None,
) -> Sequence:
    """A main (targeting "s0", kind rq1) or robustness (targeting "independent", kind
    independent) sequence.

    The PROFILES[profile] counts in random order ("order" stream); each event is sampled
    against the instance after all earlier events (current d_s, current availability).
    k = scale, or instance.meta.scale if None. With "independent", s0 is ignored.
    ValueError if horizon != sum of the counts or targeting is unknown.
    """
    if targeting not in TARGETING:
        raise ValueError(f"unknown targeting: {targeting!r}")

    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile!r}")

    counts = PROFILES[profile]

    if horizon != sum(counts):
        raise ValueError("horizon does not match profile counts")

    k = instance.meta.scale if scale is None else scale
    _check_scale(k)

    streams = _make_streams(seed)
    order = _event_order(counts, streams["order"])
    reference = s0 if targeting == "s0" else None

    current = instance
    events = []

    for event_type in order:
        event = _sample_event(event_type, current, reference, k, streams)
        current = apply_event(current, event)
        events.append(event)

    return Sequence(
        kind=SequenceKind.RQ1 if targeting == "s0" else SequenceKind.INDEPENDENT,
        profile=profile,
        seed=seed,
        events=tuple(events),
        selected=None,
    )


def rq2_sequence(
    instance: Instance,
    s0: Schedule,
    seed: int,
    horizon: int = 15,
    scale: int | None = None,
) -> Sequence:
    """The RQ2 pattern: kind rq2, profile rq2, selected = the forced pair (w, p).

    Step 1: AvailabilityRemoval(((w, p),)), (w, p) uniform among the S0-assigned pairs
    ("selected" stream). Step 6: AvailabilityRestoration(w, p). The other steps:
    RQ2_BACKGROUND counts in random order, independent targeting, P2 with exclude=(w, p),
    P3 with exclude_volunteer=w. k = scale or instance.meta.scale (the calibrated
    rq2_background_scale is passed in by the runner). ValueError if horizon != 15.
    """
    if horizon != 15:
        raise ValueError("RQ2 requires horizon 15")

    k = instance.meta.scale if scale is None else scale
    _check_scale(k)

    streams = _make_streams(seed)

    candidates = {(w, instance.shifts[s].period) for w, s in s0}

    if not candidates:
        raise ValueError("RQ2 requires an S0-assigned pair")

    selected = _choose(candidates, 1, streams["selected"])[0]
    w, period = selected

    background = iter(_event_order(RQ2_BACKGROUND, streams["order"]))

    current = instance
    events: list[Event] = []

    for step in range(1, horizon + 1):
        event: Event
        if step == RQ2_FORCING_STEP:
            event = AvailabilityRemoval(pairs=(selected,))

        elif step == RQ2_RESTORATION_STEP:
            event = AvailabilityRestoration(
                volunteer=w,
                period=period,
            )

        else:
            event = _sample_event(
                next(background),
                current,
                None,  # Background targeting never consults S0.
                k,
                streams,
                selected=selected,
            )

        current = apply_event(current, event)
        events.append(event)

    return Sequence(
        kind=SequenceKind.RQ2,
        profile=SequenceProfile.RQ2,
        seed=seed,
        events=tuple(events),
        selected=selected,
    )
