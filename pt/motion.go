package pt

// MovingShape wraps a Shape with a time-dependent rigid transform, providing
// linear motion blur. The transform is interpolated component-wise between
// Start (at time T0) and End (at time T1); it is exact for pure translation.
// Rays carry the time at which they exist (see Camera.SetShutter) and the
// shape is positioned accordingly during intersection. Times outside
// [T0, T1] are clamped, so the shape never leaves its bounding box, which
// covers the full swept motion range.
type MovingShape struct {
	Shape
	Start, End Matrix
	T0, T1     float64
	box        Box
}

// NewMovingShape returns a shape that moves from the start transform at
// time t0 to the end transform at time t1. If t0 == t1 the shape is frozen
// at the start transform.
func NewMovingShape(shape Shape, start, end Matrix, t0, t1 float64) Shape {
	s := MovingShape{}
	s.Shape = shape
	s.Start = start
	s.End = end
	s.T0 = t0
	s.T1 = t1
	inner := shape.BoundingBox()
	s.box = start.MulBox(inner).Extend(end.MulBox(inner))
	return &s
}

// NewMovingSphere returns a sphere of the given radius that moves linearly
// from center c0 at time t0 to center c1 at time t1.
func NewMovingSphere(c0, c1 Vector, radius float64, material Material, t0, t1 float64) Shape {
	sphere := NewSphere(Vector{}, radius, material)
	return NewMovingShape(sphere, Translate(c0), Translate(c1), t0, t1)
}

// matrixAt returns the interpolated transform at the given time, clamped to
// the [T0, T1] interval.
func (s *MovingShape) matrixAt(time float64) Matrix {
	alpha := 0.0
	if s.T1 != s.T0 {
		alpha = (time - s.T0) / (s.T1 - s.T0)
		alpha = Clamp(alpha, 0, 1)
	}
	return lerpMatrix(s.Start, s.End, alpha)
}

func lerpMatrix(a, b Matrix, t float64) Matrix {
	u := 1 - t
	return Matrix{
		a.x00*u + b.x00*t, a.x01*u + b.x01*t, a.x02*u + b.x02*t, a.x03*u + b.x03*t,
		a.x10*u + b.x10*t, a.x11*u + b.x11*t, a.x12*u + b.x12*t, a.x13*u + b.x13*t,
		a.x20*u + b.x20*t, a.x21*u + b.x21*t, a.x22*u + b.x22*t, a.x23*u + b.x23*t,
		a.x30*u + b.x30*t, a.x31*u + b.x31*t, a.x32*u + b.x32*t, a.x33*u + b.x33*t,
	}
}

func (s *MovingShape) Compile() {
	s.Shape.Compile()
}

func (s *MovingShape) BoundingBox() Box {
	return s.box
}

func (s *MovingShape) Intersect(r Ray) Hit {
	m := s.matrixAt(r.Time)
	inv := m.Inverse()
	shapeRay := inv.MulRay(r)
	hit := s.Shape.Intersect(shapeRay)
	if !hit.Ok() {
		return hit
	}
	shape := hit.Shape
	shapePosition := shapeRay.Position(hit.T)
	shapeNormal := shape.NormalAt(shapePosition)
	position := m.MulPosition(shapePosition)
	normal := inv.Transpose().MulDirection(shapeNormal)
	material := MaterialAt(shape, shapePosition)
	inside := false
	if shapeNormal.Dot(shapeRay.Direction) > 0 {
		normal = normal.Negate()
		inside = true
	}
	ray := Ray{position, normal, r.Time}
	info := HitInfo{shape, position, normal, ray, material, inside}
	hit.T = position.Sub(r.Origin).Length()
	hit.HitInfo = &info
	return hit
}
