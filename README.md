compute dense features for entire image (seamless)

extract vertxs

## feature

```typescript
export interface Point2D {
  x: number
  y: number
}

export interface Point3D {
  x: number
  y: number
  z: number
}

export interface Vertex {
  id: string
  position: Point2D
  position3D?: Point3D
  grounded: boolean
  tileId: string
}

export interface Edge {
  id: string
  vertexIds: string[]
  vertical: boolean
  horizontal: boolean
  length: number
  tileId: string
}

export interface Face {
  id: string
  vertexIds: string[]
  edgeIds: string[]
  isRoof: boolean
  isWall: boolean
  tileId: string
}

export interface Building {
  id: string
  vertices: Record<string, Vertex>
  edges: Record<string, Edge>
  faces: Record<string, Face>
  tileIds: string[]
}
```