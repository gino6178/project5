"""Canonical furniture taxonomy and functional priors.

3D-FRONT/3D-FUTURE, SAGE-10k and the procedural generator all speak different
category strings.  Everything is folded into one canonical vocabulary here, and
every canonical category carries the functional priors the retargeting energies
need: whether it wants a wall, how much walking room it needs in front, how
strongly it anchors a room, and how freely it may be dropped when the target
room shrinks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "CategoryPrior", "PRIORS", "canonical_category", "prior",
    "CORE_CATEGORIES", "DECOR_CATEGORIES", "SEATING_CATEGORIES",
    "ROOM_TYPES", "canonical_room_type",
]


@dataclass(frozen=True)
class CategoryPrior:
    """Functional priors for one canonical category.

    Attributes
    ----------
    wall:        affinity for being pushed against a wall, in [0, 1].
    front_clear: metres of free floor the object needs in front of it.
    side_clear:  metres of free floor it needs on its left/right.
    anchor:      how strongly it defines the room's function, in [0, 1].
                 Feeds zeta^semantic in eq. (26).
    droppable:   how acceptable it is to delete when space runs out, in [0, 1].
    multiplicity: typical repeat count; >1 marks objects that come in groups
                 and can therefore be *structurally* pruned (4 chairs -> 2).
    on_support:  the object normally rests on another object, not the floor.
    """

    wall: float = 0.3
    front_clear: float = 0.3
    side_clear: float = 0.05
    anchor: float = 0.3
    droppable: float = 0.5
    multiplicity: int = 1
    on_support: bool = False


# canonical name -> prior
PRIORS: dict[str, CategoryPrior] = {
    # ---- sleeping ----
    "double_bed":     CategoryPrior(wall=0.95, front_clear=0.55, side_clear=0.45, anchor=1.00, droppable=0.00),
    "single_bed":     CategoryPrior(wall=0.95, front_clear=0.50, side_clear=0.35, anchor=0.95, droppable=0.05),
    "kids_bed":       CategoryPrior(wall=0.95, front_clear=0.45, side_clear=0.30, anchor=0.90, droppable=0.05),
    "bunk_bed":       CategoryPrior(wall=0.95, front_clear=0.45, side_clear=0.30, anchor=0.90, droppable=0.05),
    "nightstand":     CategoryPrior(wall=0.80, front_clear=0.25, anchor=0.35, droppable=0.55, multiplicity=2),
    # ---- storage ----
    "wardrobe":       CategoryPrior(wall=0.98, front_clear=0.65, anchor=0.70, droppable=0.25),
    "cabinet":        CategoryPrior(wall=0.90, front_clear=0.50, anchor=0.35, droppable=0.55),
    "shelf":          CategoryPrior(wall=0.92, front_clear=0.45, anchor=0.30, droppable=0.60),
    "bookcase":       CategoryPrior(wall=0.95, front_clear=0.50, anchor=0.40, droppable=0.50),
    "sideboard":      CategoryPrior(wall=0.92, front_clear=0.45, anchor=0.35, droppable=0.55),
    "drawer_chest":   CategoryPrior(wall=0.90, front_clear=0.55, anchor=0.35, droppable=0.55),
    "wine_cabinet":   CategoryPrior(wall=0.90, front_clear=0.45, anchor=0.25, droppable=0.70),
    "tv_stand":       CategoryPrior(wall=0.95, front_clear=0.35, anchor=0.65, droppable=0.25),
    "shoe_cabinet":   CategoryPrior(wall=0.92, front_clear=0.40, anchor=0.20, droppable=0.75),
    # ---- seating ----
    "sofa":           CategoryPrior(wall=0.70, front_clear=0.55, side_clear=0.10, anchor=1.00, droppable=0.00),
    "l_sofa":         CategoryPrior(wall=0.75, front_clear=0.55, side_clear=0.10, anchor=1.00, droppable=0.00),
    "loveseat":       CategoryPrior(wall=0.65, front_clear=0.50, anchor=0.75, droppable=0.25),
    "armchair":       CategoryPrior(wall=0.35, front_clear=0.45, anchor=0.50, droppable=0.55, multiplicity=2),
    "lounge_chair":   CategoryPrior(wall=0.30, front_clear=0.45, anchor=0.40, droppable=0.60, multiplicity=2),
    "dining_chair":   CategoryPrior(wall=0.05, front_clear=0.50, side_clear=0.05, anchor=0.55, droppable=0.45, multiplicity=4),
    "office_chair":   CategoryPrior(wall=0.05, front_clear=0.45, anchor=0.40, droppable=0.55),
    "stool":          CategoryPrior(wall=0.15, front_clear=0.30, anchor=0.20, droppable=0.80, multiplicity=2),
    "barstool":       CategoryPrior(wall=0.10, front_clear=0.35, anchor=0.25, droppable=0.75, multiplicity=2),
    "bench":          CategoryPrior(wall=0.55, front_clear=0.40, anchor=0.30, droppable=0.65),
    # ---- tables ----
    "dining_table":   CategoryPrior(wall=0.05, front_clear=0.60, side_clear=0.60, anchor=1.00, droppable=0.00),
    "coffee_table":   CategoryPrior(wall=0.05, front_clear=0.30, side_clear=0.30, anchor=0.70, droppable=0.30),
    "side_table":     CategoryPrior(wall=0.35, front_clear=0.25, anchor=0.25, droppable=0.75, multiplicity=2),
    "desk":           CategoryPrior(wall=0.80, front_clear=0.70, anchor=0.60, droppable=0.35),
    "dressing_table": CategoryPrior(wall=0.85, front_clear=0.65, anchor=0.45, droppable=0.50),
    "console_table":  CategoryPrior(wall=0.90, front_clear=0.40, anchor=0.20, droppable=0.75),
    # ---- appliance / fixture ----
    "tv":             CategoryPrior(wall=0.85, front_clear=0.20, anchor=0.70, droppable=0.20, on_support=True),
    "fireplace":      CategoryPrior(wall=0.98, front_clear=0.60, anchor=0.55, droppable=0.35),
    "piano":          CategoryPrior(wall=0.70, front_clear=0.70, anchor=0.55, droppable=0.40),
    # ---- lighting / decor ----
    "floor_lamp":     CategoryPrior(wall=0.40, front_clear=0.15, anchor=0.15, droppable=0.85),
    "table_lamp":     CategoryPrior(wall=0.20, front_clear=0.05, anchor=0.05, droppable=0.95, on_support=True),
    "pendant_lamp":   CategoryPrior(wall=0.00, front_clear=0.00, anchor=0.10, droppable=0.80),
    "ceiling_lamp":   CategoryPrior(wall=0.00, front_clear=0.00, anchor=0.10, droppable=0.80),
    "rug":            CategoryPrior(wall=0.05, front_clear=0.00, anchor=0.20, droppable=0.80),
    "plant":          CategoryPrior(wall=0.55, front_clear=0.10, anchor=0.05, droppable=0.95, multiplicity=2),
    "decoration":     CategoryPrior(wall=0.45, front_clear=0.05, anchor=0.05, droppable=1.00, multiplicity=3),
    "wall_art":       CategoryPrior(wall=1.00, front_clear=0.00, anchor=0.10, droppable=0.90, on_support=True),
    "mirror":         CategoryPrior(wall=0.95, front_clear=0.30, anchor=0.15, droppable=0.85),
    "misc":           CategoryPrior(),
}

CORE_CATEGORIES = frozenset(
    c for c, p in PRIORS.items() if p.anchor >= 0.6
)
DECOR_CATEGORIES = frozenset(
    c for c, p in PRIORS.items() if p.droppable >= 0.8
)
SEATING_CATEGORIES = frozenset({
    "sofa", "l_sofa", "loveseat", "armchair", "lounge_chair", "dining_chair",
    "office_chair", "stool", "barstool", "bench",
})

# Raw substring -> canonical.  Ordered: the first match wins, so put the
# specific patterns before the generic ones.
_RULES: list[tuple[str, str]] = [
    (r"king.?size bed|double bed|queen", "double_bed"),
    (r"bunk bed", "bunk_bed"),
    (r"kids? bed|children.*bed", "kids_bed"),
    (r"single bed", "single_bed"),
    (r"\bbed frame\b|\bbed\b", "double_bed"),
    (r"nightstand|night stand|bedside", "nightstand"),
    (r"wardrobe|closet|armoire", "wardrobe"),
    (r"bookcase|bookshelf|book shelf", "bookcase"),
    (r"tv stand|tv cabinet|tv unit|media console|tv_stand", "tv_stand"),
    (r"shoe cabinet|shoe rack", "shoe_cabinet"),
    (r"wine cabinet|wine cooler|bar cabinet", "wine_cabinet"),
    (r"sideboard|side cabinet|console table|buffet", "sideboard"),
    (r"drawer chest|chest of drawers|corner cabinet|dresser", "drawer_chest"),
    (r"children cabinet|cabinet|cupboard|storage unit", "cabinet"),
    (r"\bshelf\b|shelving|rack", "shelf"),
    (r"l.?shaped sofa|sectional|corner sofa|chaise", "l_sofa"),
    (r"loveseat|two.?seat sofa", "loveseat"),
    (r"three.?seat|multi.?seat|lazy sofa|\bsofa\b|couch|settee", "sofa"),
    (r"armchair|arm chair", "armchair"),
    (r"dining chair|classic chinese chair|dressing chair", "dining_chair"),
    (r"office chair|desk chair", "office_chair"),
    (r"lounge chair|cafe chair|accent chair|\bchair\b", "lounge_chair"),
    (r"barstool|bar stool|counter stool", "barstool"),
    (r"footstool|sofastool|bed end stool|ottoman|pouf|\bstool\b", "stool"),
    (r"\bbench\b", "bench"),
    (r"dining table|dinner table", "dining_table"),
    (r"coffee table|cocktail table|tea table", "coffee_table"),
    (r"corner.?side table|round end table|end table|side table|nesting table", "side_table"),
    (r"dressing table|vanity", "dressing_table"),
    (r"\bdesk\b|writing table|computer table|study table", "desk"),
    (r"\btable\b", "dining_table"),
    (r"\btv\b|television|monitor|screen", "tv"),
    (r"fireplace", "fireplace"),
    (r"piano", "piano"),
    (r"floor lamp|standing lamp|torchiere", "floor_lamp"),
    (r"table lamp|desk lamp|night lamp", "table_lamp"),
    (r"pendant lamp|pendant light|chandelier", "pendant_lamp"),
    (r"ceiling lamp|ceiling light|flush mount", "ceiling_lamp"),
    (r"\brug\b|carpet|mat\b", "rug"),
    (r"plant|tree|flower|bonsai|planter", "plant"),
    (r"painting|picture|poster|wall art|artwork|frame", "wall_art"),
    (r"mirror", "mirror"),
    (r"lantern", "pendant_lamp"),
    (r"coat ?stand|coat ?rack|hat ?stand|umbrella stand", "shelf"),
    (r"\bseat\b|banquette", "lounge_chair"),
    (r"vase|sculpture|ornament|figurine|book\b|clock|basket|bowl|candle|"
     r"decorat|\bdecor\b|accessor|cushion|pillow|blanket|towel|toy|bottle|"
     r"cup|mug|tray|\bpen\b|notebook|magazine|plate|jar|bin\b|box\b|"
     r"speaker|remote|laptop|keyboard|phone|game|statue|globe|photo", "decoration"),
]
_COMPILED = [(re.compile(p, re.I), c) for p, c in _RULES]

_SUPER_FALLBACK = {
    "bed": "double_bed", "chair": "lounge_chair", "sofa": "sofa",
    "table": "dining_table", "lighting": "pendant_lamp",
    "pier/stool": "stool", "cabinet/shelf/desk": "cabinet",
    "others": "decoration",
}


def canonical_category(raw: str | None, super_category: str | None = None) -> str:
    """Map a dataset-native category string onto the canonical vocabulary."""
    if raw:
        text = str(raw).replace("_", " ").replace("/", " / ").strip()
        for pat, canon in _COMPILED:
            if pat.search(text):
                return canon
    if super_category:
        sc = str(super_category).strip().lower()
        if sc in _SUPER_FALLBACK:
            return _SUPER_FALLBACK[sc]
        for pat, canon in _COMPILED:
            if pat.search(sc):
                return canon
    return "misc"


def prior(category: str) -> CategoryPrior:
    return PRIORS.get(category, PRIORS["misc"])


ROOM_TYPES = ("bedroom", "living_room", "dining_room", "library", "other")

_ROOM_RULES = [
    (r"master.?bedroom|second.?bedroom|kids.?room|elder.?room|nanny.?room|bedroom|sleeping", "bedroom"),
    (r"living.?dining|livingdining", "living_room"),
    (r"living.?room|lounge|salon|family.?room", "living_room"),
    (r"dining.?room|dining|kitchen.?dining", "dining_room"),
    (r"library|study|office|reading", "library"),
]
_ROOM_COMPILED = [(re.compile(p, re.I), c) for p, c in _ROOM_RULES]


def canonical_room_type(raw: str | None) -> str:
    if not raw:
        return "other"
    text = str(raw).replace("_", " ")
    for pat, canon in _ROOM_COMPILED:
        if pat.search(text):
            return canon
    return "other"
