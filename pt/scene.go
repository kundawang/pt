package pt

import "sync/atomic"

type Scene struct {
	Color        Color
	Texture      Texture
	TextureAngle float64
	Shapes       []Shape
	Lights       []Shape
	tree         *Tree
	// rays counts every Scene.Intersect call (primary + secondary).
	// It is the same counter historically exposed via RayCount.
	rays uint64
	// secondaryRays counts intersection tests that are known to be
	// non-primary (bounce and shadow rays). Primary rays are accounted
	// for by the renderer; the split is stored here so that a snapshot
	// taken straight from the scene is still internally consistent.
	secondaryRays uint64
}

func (s *Scene) Compile() {
	for _, shape := range s.Shapes {
		shape.Compile()
	}
	if s.tree == nil {
		// NewTree tolerates an empty shape slice and yields a tree
		// whose Intersect always returns NoHit, so empty scenes render
		// to the background color without any nil dereference.
		s.tree = NewTree(s.Shapes)
	}
}

func (s *Scene) Add(shape Shape) {
	s.Shapes = append(s.Shapes, shape)
	if shape.MaterialAt(Vector{}).Emittance > 0 {
		s.Lights = append(s.Lights, shape)
	}
}

// RayCount returns the total number of intersection tests performed
// against the scene since the last ResetRays. Safe for concurrent use.
func (s *Scene) RayCount() uint64 {
	return atomic.LoadUint64(&s.rays)
}

// SecondaryRayCount returns the number of intersection tests attributed
// to non-primary (bounce and shadow) rays. Safe for concurrent use.
func (s *Scene) SecondaryRayCount() uint64 {
	return atomic.LoadUint64(&s.secondaryRays)
}

// ResetRays zeroes the scene ray counters. It must only be called when
// no render workers are active; renderers call it at the start of an
// iteration before spawning workers.
func (s *Scene) ResetRays() {
	atomic.StoreUint64(&s.rays, 0)
	atomic.StoreUint64(&s.secondaryRays, 0)
}

// Stats returns a read-only, point-in-time snapshot of the scene's ray
// counters. The returned value contains scalars only and is safe to
// retain and read while rendering continues.
func (s *Scene) Stats() SceneStats {
	rays := atomic.LoadUint64(&s.rays)
	secondary := atomic.LoadUint64(&s.secondaryRays)
	if secondary > rays {
		// A snapshot may observe the two independent atomics at
		// slightly different instants. Never expose a negative (here,
		// inverted) primary/secondary split.
		secondary = rays
	}
	return SceneStats{Rays: rays, SecondaryRays: secondary, PrimaryRays: rays - secondary}
}

func (s *Scene) Intersect(r Ray) Hit {
	atomic.AddUint64(&s.rays, 1)
	return s.tree.Intersect(r)
}

// intersect records an intersection test and classifies it as a
// primary (camera) ray or a secondary (bounce) ray.
func (s *Scene) intersect(r Ray, primary bool) Hit {
	if !primary {
		atomic.AddUint64(&s.secondaryRays, 1)
	}
	return s.Intersect(r)
}

// intersectSecondary records a non-primary intersection test. It is
// invoked by the sampler for shadow rays.
func (s *Scene) intersectSecondary(r Ray) Hit {
	atomic.AddUint64(&s.secondaryRays, 1)
	return s.Intersect(r)
}
