package pt

import "sync/atomic"

type Scene struct {
	Color        Color
	Texture      Texture
	TextureAngle float64
	Shapes       []Shape
	Lights       []Shape
	// TreeConfig configures the spatial index built by Compile. It must be
	// set before the first Compile; change it after Invalidate to rebuild.
	TreeConfig TreeConfig
	tree       *Tree
	rays       uint64
}

// Compile prepares every shape and builds the spatial index. Repeated calls
// reuse the existing tree without rebuilding; call Invalidate after mutating
// Shapes to force a fresh build.
func (s *Scene) Compile() {
	for _, shape := range s.Shapes {
		shape.Compile()
	}
	if s.tree == nil {
		s.tree = NewTreeWithConfig(s.Shapes, s.TreeConfig)
	}
}

// Invalidate discards the compiled spatial index so the next Compile rebuilds
// it. The index itself is immutable once built, so concurrent read-only
// Intersect calls share it without synchronization.
func (s *Scene) Invalidate() {
	s.tree = nil
}

// TreeStats returns build statistics for the compiled tree, or the zero value
// before Compile.
func (s *Scene) TreeStats() TreeStats {
	if s.tree == nil {
		return TreeStats{}
	}
	return s.tree.Stats()
}

// TreeCandidates returns shape intersection tests counted since the last
// reset. The counter is active only when TreeConfig.CountCandidates is set.
func (s *Scene) TreeCandidates() uint64 {
	if s.tree == nil {
		return 0
	}
	return s.tree.Candidates()
}

// ResetTreeCandidates zeroes the runtime shape-test counter.
func (s *Scene) ResetTreeCandidates() {
	if s.tree != nil {
		s.tree.ResetCandidates()
	}
}

func (s *Scene) Add(shape Shape) {
	s.Shapes = append(s.Shapes, shape)
	// Any previously built index no longer covers the complete scene.
	s.Invalidate()
	if shape.MaterialAt(Vector{}).Emittance > 0 {
		s.Lights = append(s.Lights, shape)
	}
}

func (s *Scene) RayCount() uint64 {
	return atomic.LoadUint64(&s.rays)
}

func (s *Scene) Intersect(r Ray) Hit {
	atomic.AddUint64(&s.rays, 1)
	return s.tree.Intersect(r)
}
