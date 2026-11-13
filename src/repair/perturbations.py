"""Events: how an instance changes between steps (exposé 4.4).

apply_event never mutates: it returns a new Instance (Instance.with_volunteer /
with_shift). For now only AvailabilityRemoval; the other event types raise
NotImplementedError until they are needed.
"""

import copy

from dataclasses import replace
from repair.model import AvailabilityRemoval, PreferenceChange, Event, Instance


def apply_event(instance: Instance, event: Event) -> Instance:
    """The instance after event.

    AvailabilityRemoval(pairs): for each (volunteer, period), remove period from
    that volunteer's available set. Everything else stays the same.
    """
    if isinstance(event, AvailabilityRemoval):
        # AvailabilityRemoval(pairs=((0, 0),))      # volunteer 0 loses period 0
        # AvailabilityRemoval(pairs=((0, 0), (2, 1)))  # several at once is possible
        for w, p in event.pairs:
            v = instance.volunteers[w]
            available = replace(v, available=v.available - {p})

            instance = instance.with_volunteer(available)

        return instance

    if isinstance(event, PreferenceChange):
        for w, r, new in event.changes:
            v = instance.volunteers[w]
            new_pref_role = list(copy.deepcopy(v.pref_role))
            new_pref_role[r] = new

            new_volunteer = replace(v, pref_role=tuple(new_pref_role))

            instance = instance.with_volunteer(new_volunteer)
        return instance
    raise NotImplementedError(f"{type(event).__name__} is not supported yet")
