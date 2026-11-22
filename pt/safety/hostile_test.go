package safety

import (
	"math"

	"github.com/fogleman/pt/pt"
)

// Hostile SDF test doubles. They implement pt.SDF directly so the tests can
// prove the guard layer works even when the underlying SDF is adversarial.

// nanSDF always returns NaN. The upstream pt.SDFShape.Intersect computes
// t += NaN forever (only bounded by its hardcoded 1000-iteration cap) and can
// never satisfy d < epsilon; the bounded marcher must terminate immediately.
type nanSDF struct{}

func (nanSDF) Evaluate(pt.Vector) float64 { return math.NaN() }
func (nanSDF) BoundingBox() pt.Box {
	return pt.Box{Min: pt.Vector{X: -1, Y: -1, Z: -1}, Max: pt.Vector{X: 1, Y: 1, Z: 1}}
}

// stallSDF always returns a positive distance strictly larger than the
// surface epsilon but smaller than the jump size. A naive marcher keeps
// stepping by the same amount forever without ever converging; the bounded
// marcher must hit its iteration/deadline cap (or the box boundary) and
// return NoHit instead of hanging.
type stallSDF struct{}

func (stallSDF) Evaluate(pt.Vector) float64 { return sdfEpsilon * 4 }
func (stallSDF) BoundingBox() pt.Box {
	return pt.Box{Min: pt.Vector{X: -10, Y: -10, Z: -10}, Max: pt.Vector{X: 10, Y: 10, Z: 10}}
}

// explodingSDF returns +Inf. A guarded marcher treats it as "nothing nearby"
// and exits via the box boundary; an unguarded one feeds Inf into t.
type explodingSDF struct{}

func (explodingSDF) Evaluate(pt.Vector) float64 { return math.Inf(1) }
func (explodingSDF) BoundingBox() pt.Box {
	return pt.Box{Min: pt.Vector{X: -1, Y: -1, Z: -1}, Max: pt.Vector{X: 1, Y: 1, Z: 1}}
}

// panicSDF panics on evaluation to prove the last-resort recover boundary.
type panicSDF struct{}

func (panicSDF) Evaluate(pt.Vector) float64 { panic("hostile sdf boom") }
func (panicSDF) BoundingBox() pt.Box {
	return pt.Box{Min: pt.Vector{X: -1, Y: -1, Z: -1}, Max: pt.Vector{X: 1, Y: 1, Z: 1}}
}

// recursiveSDF nests itself N levels deep. Every Evaluate descends one level;
// without GuardTree a deep enough tree exhausts the goroutine stack.
type recursiveSDF struct {
	child pt.SDF
	depth int
}

func (r *recursiveSDF) Evaluate(p pt.Vector) float64 {
	if r.child == nil {
		return 0 // leaf surface
	}
	return r.child.Evaluate(p)
}

func (r *recursiveSDF) BoundingBox() pt.Box {
	return pt.Box{Min: pt.Vector{X: -1, Y: -1, Z: -1}, Max: pt.Vector{X: 1, Y: 1, Z: 1}}
}

// buildRecursiveSDF builds a chain of `levels` nested recursiveSDF nodes.
func buildRecursiveSDF(levels int) pt.SDF {
	var root pt.SDF = &recursiveSDF{}
	for i := 1; i < levels; i++ {
		root = &recursiveSDF{child: root, depth: i}
	}
	return root
}

// unitSphere is a benign SDF used as the pixel-stability and equivalence
// baseline.
type unitSphere struct{}

func (unitSphere) Evaluate(p pt.Vector) float64 {
	return math.Sqrt(p.X*p.X+p.Y*p.Y+p.Z*p.Z) - 0.5
}
func (unitSphere) BoundingBox() pt.Box {
	return pt.Box{Min: pt.Vector{X: -1, Y: -1, Z: -1}, Max: pt.Vector{X: 1, Y: 1, Z: 1}}
}

var (
	_ pt.SDF = nanSDF{}
	_ pt.SDF = stallSDF{}
	_ pt.SDF = explodingSDF{}
	_ pt.SDF = panicSDF{}
	_ pt.SDF = unitSphere{}
)
