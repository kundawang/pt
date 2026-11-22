package safety

import (
	"math"
	"time"

	"github.com/fogleman/pt/pt"
)

// SDF parameters are untrusted in the same way mesh files are: a hostile
// Evaluate can return NaN (so the marching ray never advances) or a composite
// SDF can be nested until a single Evaluate grows the stack without bound.
// GuardedSDF and BoundedSDFShape harden both failure modes independently of
// pt.SDFShape, whose Intersect loop has no depth accounting and treats NaN as
// an ordinary (non-terminating) distance.

// depthBudget is a single counter shared by every guard node of one SDF tree.
// It is created per tree (not per goroutine) and only touched while a ray
// goroutine walks that tree, so concurrent rendering needs no synchronization
// beyond each ray owning its call stack.
type depthBudget struct {
	remaining int
}

// GuardedSDF wraps one SDF node so each descent consumes a unit of the shared
// depth budget. When the budget is exhausted Evaluate returns +Inf, which a
// marching loop treats as "advance to the box boundary" instead of recursing
// further or looping.
type GuardedSDF struct {
	Inner  pt.SDF
	limit  int
	budget *depthBudget
}

// NewGuardedSDF wraps a root SDF with a fresh depth budget. A non-positive
// limit falls back to DefaultLimits.MaxSDFDepth.
func NewGuardedSDF(sdf pt.SDF, limit int) *GuardedSDF {
	if limit <= 0 {
		limit = DefaultLimits().MaxSDFDepth
	}
	return &GuardedSDF{Inner: sdf, limit: limit, budget: &depthBudget{remaining: limit}}
}

// child wraps an inner node while sharing the root's budget.
func (g *GuardedSDF) child(sdf pt.SDF) *GuardedSDF {
	return &GuardedSDF{Inner: sdf, limit: g.limit, budget: g.budget}
}

// Limit exposes the configured depth cap for tests and reports.
func (g *GuardedSDF) Limit() int { return g.limit }

// Evaluate bounds one descent. Each nested guarded node decrements the shared
// budget; exhaustion yields +Inf rather than another recursive call.
func (g *GuardedSDF) Evaluate(p pt.Vector) float64 {
	if g.budget.remaining <= 0 {
		return math.Inf(1)
	}
	g.budget.remaining--
	restored := false
	defer func() {
		if !restored {
			g.budget.remaining++
		}
	}()
	d := g.Inner.Evaluate(p)
	g.budget.remaining++
	restored = true
	// A hostile SDF may return NaN or a negative distance with huge
	// magnitude. Normalize NaN to +Inf so the march cannot stall on t += NaN.
	if math.IsNaN(d) {
		return math.Inf(1)
	}
	return d
}

// BoundingBox forwards to the inner SDF.
func (g *GuardedSDF) BoundingBox() pt.Box { return g.Inner.BoundingBox() }

var _ pt.SDF = (*GuardedSDF)(nil)

// CheckSDFDepth independently walks an SDF tree and rejects nesting deeper
// than limits.MaxSDFDepth before any ray is evaluated. It mirrors the concrete
// composite types in pt/sdf.go and records the observed depth in the report.
// A leaf SDF has depth 1.
func CheckSDFDepth(sdf pt.SDF, limits Limits) (*Report, error) {
	limits = limits.fill()
	r := &Report{Format: "sdf"}
	depth := measureSDFDepth(sdf, 1)
	r.Vertices = depth // reuse the count field for the nesting metric
	if depth > limits.MaxSDFDepth {
		return r.fail(StageDepth, rejected(StageDepth,
			"SDF nesting depth %d exceeds limit %d", depth, limits.MaxSDFDepth))
	}
	r.Stage = StageAccepted
	return r, nil
}

// measureSDFDepth returns the maximum root-to-leaf nesting depth.
func measureSDFDepth(sdf pt.SDF, depth int) int {
	switch s := sdf.(type) {
	case *pt.UnionSDF:
		return maxChildDepth(s.Items, depth)
	case *pt.DifferenceSDF:
		return maxChildDepth(s.Items, depth)
	case *pt.IntersectionSDF:
		return maxChildDepth(s.Items, depth)
	case *pt.TransformSDF:
		return measureSDFDepth(s.SDF, depth+1)
	case *pt.ScaleSDF:
		return measureSDFDepth(s.SDF, depth+1)
	case *pt.RepeatSDF:
		return measureSDFDepth(s.SDF, depth+1)
	case *GuardedSDF:
		return measureSDFDepth(s.Inner, depth)
	default:
		return depth
	}
}

func maxChildDepth(items []pt.SDF, depth int) int {
	best := depth
	for _, item := range items {
		if d := measureSDFDepth(item, depth+1); d > best {
			best = d
		}
	}
	return best
}

// GuardTree recursively wraps every composite SDF node so depth is counted at
// every level of nesting, sharing one budget across the whole tree. It mirrors
// the concrete composite types in pt/sdf.go and leaves leaf SDFs untouched
// beyond their own wrapping node.
func GuardTree(sdf pt.SDF, limit int) pt.SDF {
	root := NewGuardedSDF(sdf, limit)
	guardChildren(sdf, root)
	return root
}

func guardChildren(sdf pt.SDF, parent *GuardedSDF) {
	switch s := sdf.(type) {
	case *pt.UnionSDF:
		for i, item := range s.Items {
			s.Items[i] = wrapChild(item, parent)
		}
	case *pt.DifferenceSDF:
		for i, item := range s.Items {
			s.Items[i] = wrapChild(item, parent)
		}
	case *pt.IntersectionSDF:
		for i, item := range s.Items {
			s.Items[i] = wrapChild(item, parent)
		}
	case *pt.TransformSDF:
		s.SDF = wrapChild(s.SDF, parent)
	case *pt.ScaleSDF:
		s.SDF = wrapChild(s.SDF, parent)
	case *pt.RepeatSDF:
		s.SDF = wrapChild(s.SDF, parent)
	}
}

func wrapChild(sdf pt.SDF, parent *GuardedSDF) pt.SDF {
	g := parent.child(sdf)
	guardChildren(sdf, g)
	return g
}

// BoundedSDFShape is an independent ray marcher that replaces pt.SDFShape for
// untrusted SDFs. It bounds iteration count, wall-clock time and per-step
// advance, and it rejects NaN distances. It implements pt.Shape so it can be
// added to a pt.Scene directly.
type BoundedSDFShape struct {
	SDF      pt.SDF
	Material pt.Material

	// MaxIterations caps march steps per ray. Zero uses a fixed safe default.
	MaxIterations int
	// Deadline bounds the wall-clock cost of one Intersect call. Zero uses
	// DefaultLimits.SampleTimeout.
	Deadline time.Duration

	// Stats are observability counters, updated after every Intersect.
	Iterations int
	Timeouts   int64 // iteration cap, deadline, NaN or panic termination
	DepthHits  int64 // +Inf / depth-cap termination
	BoxExits   int64 // clean march that left the bounding box without a hit
}

// NewBoundedSDFShape constructs a shape with conservative iteration and time
// budgets. The supplied SDF should already be wrapped with GuardTree.
func NewBoundedSDFShape(sdf pt.SDF, material pt.Material, limits Limits) *BoundedSDFShape {
	limits = limits.fill()
	return &BoundedSDFShape{
		SDF:           sdf,
		Material:      material,
		MaxIterations: 2048,
		Deadline:      limits.SampleTimeout,
	}
}

const (
	sdfEpsilon  = 1e-5
	sdfStart    = 1e-4
	sdfJumpSize = 1e-3
)

// Compile satisfies pt.Shape; SDFs need no preprocessing.
func (s *BoundedSDFShape) Compile() {}

// BoundingBox satisfies pt.Shape by forwarding to the wrapped SDF.
func (s *BoundedSDFShape) BoundingBox() pt.Box { return s.SDF.BoundingBox() }

// Intersect is a hardened re-implementation of pt.SDFShape.Intersect. Every
// exit path is finite: the loop is bounded by both an iteration counter and a
// deadline, NaN distances terminate the march, and +Inf steps jump straight
// to the box exit. It never panics on degenerate SDF output.
func (s *BoundedSDFShape) Intersect(ray pt.Ray) (hit pt.Hit) {
	defer func() {
		// Last-resort containment: even an unguarded panic inside a hostile
		// Evaluate must surface as NoHit rather than killing a render worker.
		if r := recover(); r != nil {
			s.Timeouts++
			hit = pt.NoHit
		}
	}()

	maxIter := s.MaxIterations
	if maxIter <= 0 {
		maxIter = 2048
	}
	deadline := s.Deadline
	if deadline <= 0 {
		deadline = DefaultLimits().SampleTimeout
	}
	stop := time.Now().Add(deadline)

	box := s.SDF.BoundingBox()
	t1, t2 := box.Intersect(ray)
	if t2 < t1 || t2 < 0 || math.IsNaN(t1) || math.IsNaN(t2) || math.IsInf(t2, 0) {
		return pt.NoHit
	}
	t := math.Max(sdfStart, t1)
	if math.IsNaN(t) || math.IsInf(t, 0) {
		return pt.NoHit
	}
	jump := true
	for i := 0; i < maxIter; i++ {
		if i&63 == 63 && time.Now().After(stop) {
			s.Timeouts++
			return pt.NoHit
		}
		d := s.SDF.Evaluate(ray.Position(t))
		// NaN cannot advance t; +Inf means "nothing nearby", jump to exit.
		if math.IsNaN(d) {
			s.Timeouts++
			return pt.NoHit
		}
		if math.IsInf(d, 1) {
			s.DepthHits++
			return pt.NoHit
		}
		if jump && d < 0 {
			t -= sdfJumpSize
			jump = false
			if t < 0 {
				return pt.NoHit
			}
			continue
		}
		if d < sdfEpsilon {
			s.Iterations += i + 1
			return pt.Hit{Shape: s, T: t}
		}
		if jump && d < sdfJumpSize {
			d = sdfJumpSize
		}
		if d <= 0 {
			// Defensive: a non-NaN, non-Inf zero/negative stall distance must
			// not pin the march at one t forever.
			d = sdfJumpSize
		}
		t += d
		if t > t2 {
			s.Iterations += i + 1
			s.BoxExits++
			return pt.NoHit
		}
	}
	s.Timeouts++
	return pt.NoHit
}

// UV satisfies pt.Shape; SDFs carry no intrinsic UVs.
func (s *BoundedSDFShape) UV(p pt.Vector) pt.Vector { return pt.Vector{} }

// NormalAt estimates the normal with central differences, matching
// pt.SDFShape but guarded against non-finite output.
func (s *BoundedSDFShape) NormalAt(p pt.Vector) pt.Vector {
	const e = 1e-4
	n := pt.Vector{
		X: s.SDF.Evaluate(pt.Vector{X: p.X - e, Y: p.Y, Z: p.Z}) - s.SDF.Evaluate(pt.Vector{X: p.X + e, Y: p.Y, Z: p.Z}),
		Y: s.SDF.Evaluate(pt.Vector{X: p.X, Y: p.Y - e, Z: p.Z}) - s.SDF.Evaluate(pt.Vector{X: p.X, Y: p.Y + e, Z: p.Z}),
		Z: s.SDF.Evaluate(pt.Vector{X: p.X, Y: p.Y, Z: p.Z - e}) - s.SDF.Evaluate(pt.Vector{X: p.X, Y: p.Y, Z: p.Z + e}),
	}
	length := n.Length()
	if length == 0 || math.IsNaN(length) || math.IsInf(length, 1) {
		return pt.Vector{X: 0, Y: 0, Z: 1}
	}
	return n.Normalize()
}

// MaterialAt satisfies pt.Shape.
func (s *BoundedSDFShape) MaterialAt(pt.Vector) pt.Material { return s.Material }

var _ pt.Shape = (*BoundedSDFShape)(nil)
