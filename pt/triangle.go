package pt

import "math"

type Triangle struct {
	Material   *Material
	V1, V2, V3 Vector
	N1, N2, N3 Vector
	T1, T2, T3 Vector
	C1, C2, C3 Color
	// c1-c3 record whether a vertex color was configured. The zero Color is
	// a valid black vertex color, so presence cannot be inferred from value.
	c1, c2, c3 bool
}

func NewTriangle(v1, v2, v3, t1, t2, t3 Vector, material Material) *Triangle {
	t := Triangle{}
	t.V1 = v1
	t.V2 = v2
	t.V3 = v3
	t.T1 = t1
	t.T2 = t2
	t.T3 = t3
	t.Material = &material
	t.FixNormals()
	return &t
}

func (t *Triangle) Vertices() (Vector, Vector, Vector) {
	return t.V1, t.V2, t.V3
}

// sanitizeColor guarantees a finite color in [0, 1]. Non-finite components
// (NaN, +Inf, -Inf) become 0 and out-of-range components are clamped, so bad
// input can never propagate NaN into the renderer.
func sanitizeColor(c Color) Color {
	if math.IsNaN(c.R) || math.IsInf(c.R, 0) {
		c.R = 0
	}
	if math.IsNaN(c.G) || math.IsInf(c.G, 0) {
		c.G = 0
	}
	if math.IsNaN(c.B) || math.IsInf(c.B, 0) {
		c.B = 0
	}
	return Color{Clamp(c.R, 0, 1), Clamp(c.G, 0, 1), Clamp(c.B, 0, 1)}
}

// SetVertexColors configures per-vertex colors, one per triangle corner, in
// the same order as V1, V2, V3. The colors are interpolated at a hit point
// using its barycentric coordinates. Values are sanitized (non-finite to 0,
// out-of-range clamped to [0, 1]) before being stored.
func (t *Triangle) SetVertexColors(c1, c2, c3 Color) {
	t.C1, t.C2, t.C3 = sanitizeColor(c1), sanitizeColor(c2), sanitizeColor(c3)
	t.c1, t.c2, t.c3 = true, true, true
}

// SetVertexColor configures the color of a single vertex: index 0, 1 or 2
// corresponds to V1, V2, V3. Other indices are ignored. Vertices without a
// configured color fall back to the material color during interpolation.
func (t *Triangle) SetVertexColor(index int, c Color) {
	c = sanitizeColor(c)
	switch index {
	case 0:
		t.C1, t.c1 = c, true
	case 1:
		t.C2, t.c2 = c, true
	case 2:
		t.C3, t.c3 = c, true
	}
}

// ClearVertexColors removes any configured per-vertex colors, restoring the
// material/texture-only behavior.
func (t *Triangle) ClearVertexColors() {
	t.C1, t.C2, t.C3 = Color{}, Color{}, Color{}
	t.c1, t.c2, t.c3 = false, false, false
}

// HasVertexColors reports whether at least one vertex color is configured.
func (t *Triangle) HasVertexColors() bool {
	return t.c1 || t.c2 || t.c3
}

// VertexColorAt returns the per-vertex color interpolated with the
// barycentric coordinates of p, in linear color space. Vertex colors are
// attached to vertices rather than positions, so perspective projections,
// triangle transforms and sharing a triangle in a mesh leave the ordering
// untouched. Vertices without an explicit color use the material color.
// Degenerate triangles (whose barycentric coordinates are undefined) get a
// finite equal-weight average instead of NaN.
func (t *Triangle) VertexColorAt(p Vector) Color {
	u, v, w := t.Barycentric(p)
	base := t.Material.Color
	c1, c2, c3 := base, base, base
	if t.c1 {
		c1 = t.C1
	}
	if t.c2 {
		c2 = t.C2
	}
	if t.c3 {
		c3 = t.C3
	}
	// Barycentric weights are undefined for a degenerate triangle (division
	// by zero). Fall back to an equal-weight average, which is always finite.
	if math.IsNaN(u) || math.IsInf(u, 0) || math.IsNaN(v) || math.IsInf(v, 0) ||
		math.IsNaN(w) || math.IsInf(w, 0) {
		u, v, w = 1.0/3.0, 1.0/3.0, 1.0/3.0
	}
	c := Color{}
	c = c.Add(c1.MulScalar(u))
	c = c.Add(c2.MulScalar(v))
	c = c.Add(c3.MulScalar(w))
	// Defense in depth: stored colors are already sanitized, this keeps any
	// unforeseen non-finite arithmetic from escaping into the sampler.
	return sanitizeColor(c)
}

func (t *Triangle) Compile() {
}

func (t *Triangle) BoundingBox() Box {
	min := t.V1.Min(t.V2).Min(t.V3)
	max := t.V1.Max(t.V2).Max(t.V3)
	return Box{min, max}
}

func (t *Triangle) Intersect(r Ray) Hit {
	e1x := t.V2.X - t.V1.X
	e1y := t.V2.Y - t.V1.Y
	e1z := t.V2.Z - t.V1.Z
	e2x := t.V3.X - t.V1.X
	e2y := t.V3.Y - t.V1.Y
	e2z := t.V3.Z - t.V1.Z
	px := r.Direction.Y*e2z - r.Direction.Z*e2y
	py := r.Direction.Z*e2x - r.Direction.X*e2z
	pz := r.Direction.X*e2y - r.Direction.Y*e2x
	det := e1x*px + e1y*py + e1z*pz
	if det > -EPS && det < EPS {
		return NoHit
	}
	inv := 1 / det
	tx := r.Origin.X - t.V1.X
	ty := r.Origin.Y - t.V1.Y
	tz := r.Origin.Z - t.V1.Z
	u := (tx*px + ty*py + tz*pz) * inv
	if u < 0 || u > 1 {
		return NoHit
	}
	qx := ty*e1z - tz*e1y
	qy := tz*e1x - tx*e1z
	qz := tx*e1y - ty*e1x
	v := (r.Direction.X*qx + r.Direction.Y*qy + r.Direction.Z*qz) * inv
	if v < 0 || u+v > 1 {
		return NoHit
	}
	d := (e2x*qx + e2y*qy + e2z*qz) * inv
	if d < EPS {
		return NoHit
	}
	return Hit{t, d, nil}
}

func (t *Triangle) UV(p Vector) Vector {
	u, v, w := t.Barycentric(p)
	n := Vector{}
	n = n.Add(t.T1.MulScalar(u))
	n = n.Add(t.T2.MulScalar(v))
	n = n.Add(t.T3.MulScalar(w))
	return Vector{n.X, n.Y, 0}
}

func (t *Triangle) MaterialAt(p Vector) Material {
	return *t.Material
}

func (t *Triangle) NormalAt(p Vector) Vector {
	u, v, w := t.Barycentric(p)
	n := Vector{}
	n = n.Add(t.N1.MulScalar(u))
	n = n.Add(t.N2.MulScalar(v))
	n = n.Add(t.N3.MulScalar(w))
	n = n.Normalize()
	if t.Material.NormalTexture != nil {
		b := Vector{}
		b = b.Add(t.T1.MulScalar(u))
		b = b.Add(t.T2.MulScalar(v))
		b = b.Add(t.T3.MulScalar(w))
		ns := t.Material.NormalTexture.NormalSample(b.X, b.Y)
		dv1 := t.V2.Sub(t.V1)
		dv2 := t.V3.Sub(t.V1)
		dt1 := t.T2.Sub(t.T1)
		dt2 := t.T3.Sub(t.T1)
		T := dv1.MulScalar(dt2.Y).Sub(dv2.MulScalar(dt1.Y)).Normalize()
		B := dv2.MulScalar(dt1.X).Sub(dv1.MulScalar(dt2.X)).Normalize()
		N := T.Cross(B)
		matrix := Matrix{
			T.X, B.X, N.X, 0,
			T.Y, B.Y, N.Y, 0,
			T.Z, B.Z, N.Z, 0,
			0, 0, 0, 1}
		n = matrix.MulDirection(ns)
	}
	if t.Material.BumpTexture != nil {
		b := Vector{}
		b = b.Add(t.T1.MulScalar(u))
		b = b.Add(t.T2.MulScalar(v))
		b = b.Add(t.T3.MulScalar(w))
		bump := t.Material.BumpTexture.BumpSample(b.X, b.Y)
		dv1 := t.V2.Sub(t.V1)
		dv2 := t.V3.Sub(t.V1)
		dt1 := t.T2.Sub(t.T1)
		dt2 := t.T3.Sub(t.T1)
		tangent := dv1.MulScalar(dt2.Y).Sub(dv2.MulScalar(dt1.Y)).Normalize()
		bitangent := dv2.MulScalar(dt1.X).Sub(dv1.MulScalar(dt2.X)).Normalize()
		n = n.Add(tangent.MulScalar(bump.X * t.Material.BumpMultiplier))
		n = n.Add(bitangent.MulScalar(bump.Y * t.Material.BumpMultiplier))
	}
	n = n.Normalize()
	return n
}

func (t *Triangle) Area() float64 {
	e1 := t.V2.Sub(t.V1)
	e2 := t.V3.Sub(t.V1)
	n := e1.Cross(e2)
	return n.Length() / 2
}

func (t *Triangle) Barycentric(p Vector) (u, v, w float64) {
	v0 := t.V2.Sub(t.V1)
	v1 := t.V3.Sub(t.V1)
	v2 := p.Sub(t.V1)
	d00 := v0.Dot(v0)
	d01 := v0.Dot(v1)
	d11 := v1.Dot(v1)
	d20 := v2.Dot(v0)
	d21 := v2.Dot(v1)
	d := d00*d11 - d01*d01
	v = (d11*d20 - d01*d21) / d
	w = (d00*d21 - d01*d20) / d
	u = 1 - v - w
	return
}

func (t *Triangle) FixNormals() {
	e1 := t.V2.Sub(t.V1)
	e2 := t.V3.Sub(t.V1)
	n := e1.Cross(e2).Normalize()
	zero := Vector{}
	if t.N1 == zero {
		t.N1 = n
	}
	if t.N2 == zero {
		t.N2 = n
	}
	if t.N3 == zero {
		t.N3 = n
	}
}
