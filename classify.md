# Classify

Each pixel in the map image is classified into exactly one of the categories below.
Categories are grouped by theme for readability; the classifier treats them as a flat
set of labels.

## Off-map / annotation

- `border` — pixels outside the map area (margins, frames, off-map background).
- `label` — text and cartographic annotations drawn on the map.

## Roads and ground surfaces

- `road` — road.
- `dirt` — bare earth, mud, or unvegetated ground that is not a road.

## Water

- `water` — any body of water (river, stream, pond, canal, fountain basin, etc.).

## Vegetation

- `field` — cultivated crops growing in a managed plot.
- `garden` — ornamental planting or flower beds.
- `lawn` — manicured grass.
- `tree` — tall woody plant with a canopy.
- `bush` — short woody plant, not part of a field.
- `hedge` — wall made of plants.

## Structures and built features

- `building` — generic built structure pixels that are not more specifically
  labeled as `roof`, `wall`, `window`, `door`, or `chimney`.
- `roof` — the top surface of a building. Use a subtype when the shape is clear;
  fall back to plain `roof` otherwise. Subtypes reflect roof forms common in
  1790s Paris:
  - `roof_gable` — two sloping sides meeting at a ridge, with a triangular
    gable wall at each end (*toit à deux pentes / pignon*).
  - `roof_hip` — slopes on all four sides, no gable wall (*toit en croupe*).
  - `roof_mansard` — double-pitched roof with a steep lower slope and a
    shallower upper slope, often with dormers (*toit à la Mansart*).
  - `roof_shed` — single sloped plane, typical of lean-tos and outbuildings
    (*appentis*).
  - `roof_conical` — conical or pyramidal roof over a tower or corner
    pavilion (*toit en poivrière / pavillon*).
  - `roof_flat` — flat or near-flat roof surface, including rooftop terraces
    and the flat top of a fence or wall (*terrasse*).
  - `roof_dormer` — a dormer window projecting from a sloped roof
    (*lucarne*); label the dormer's own small roof with this, not the
    surrounding slope.
- `wall` — a vertical building face, or a free-standing masonry wall.
- `window` — a window opening on a building.
- `door` — a door opening on a building.
- `chimney` — a chimney on a roof.
- `windmill` — a windmill (body and/or sails).
- `bridge` — a bridge deck or span crossing water or a gap.
- `stairs` — exterior steps or staircases.
- `fence` — the side/face of a fence or wall-like barrier. The flat top cap
  of a fence or wall is labeled `roof_flat`.

## Objects

- `boat` — a boat or similar watercraft.
- `rock` — exposed rock or stone surfaces.
- `log` — a log.
