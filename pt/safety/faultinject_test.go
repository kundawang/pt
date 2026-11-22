package safety

import (
	"testing"
	"time"

	"github.com/fogleman/pt/pt"
)

// Fault injection (requirement: "用故障注入证明检查确实能捕获缺陷").
//
// Each test deliberately injects a concrete defect — either a known upstream
// loader crash or a synthetic faulty component — and proves that an
// independent checker/guard catches it at the documented stage, rather than
// relying on the defective code path to fail safely on its own. The defects
// here are the exact classes named in the task: index out of range, negative
// numbers, NaN/Inf, oversized vertices, degenerate triangles, bad STL headers
// and non-returning SDF intersection.

// injectedFault pairs a defective input with the stage that must catch it.
type injectedFault struct {
	name   string
	kind   string // "obj" | "stl"
	data   []byte
	expect Stage
}

func TestFaultInjectionCatalog(t *testing.T) {
	faults := []injectedFault{
		{
			name:   "obj index out of range (upstream pt.LoadOBJ panic)",
			kind:   "obj",
			data:   []byte("v 0 0 0\nf 1 5 9\n"),
			expect: StageIndex,
		},
		{
			name:   "obj negative index resolving out of range",
			kind:   "obj",
			data:   []byte("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -99 -98 -97\n"),
			expect: StageIndex,
		},
		{
			name:   "obj NaN coordinate",
			kind:   "obj",
			data:   []byte("v nan nan nan\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"),
			expect: StageValue,
		},
		{
			name:   "obj +Inf coordinate",
			kind:   "obj",
			data:   []byte("v inf 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"),
			expect: StageValue,
		},
		{
			name:   "obj 1e999 (overflow to Inf) coordinate",
			kind:   "obj",
			data:   []byte("v 1e999 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"),
			expect: StageValue,
		},
		{
			name:   "obj gigantic finite vertex (memory/bbox amplification)",
			kind:   "obj",
			data:   []byte("v 1e300 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"),
			expect: StageMagnitude,
		},
		{
			name:   "obj degenerate repeated-vertex face",
			kind:   "obj",
			data:   []byte("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 1 1\n"),
			expect: StageAccepted, // warned at text layer; caught geometrically below
		},
		{
			name:   "stl corrupt declared-count header (allocation amplification)",
			kind:   "stl",
			data:   genSTLBinary(newRand(1), 1, 1_000_000, true),
			expect: StageHeader,
		},
	}
	limits := DefaultLimits()
	limits.MaxCoordinate = 1e6
	for _, fault := range faults {
		t.Run(fault.name, func(t *testing.T) {
			var rep *Report
			var err error
			if fault.kind == "obj" {
				rep, err = CheckOBJBytes(fault.data, limits)
			} else {
				rep, err = CheckSTLBytes(fault.data, limits)
			}
			if fault.expect == StageAccepted {
				if err != nil {
					t.Fatalf("fault %q unexpectedly rejected at %s", fault.name, rep.Stage)
				}
				if len(rep.Warnings) == 0 {
					t.Fatalf("degenerate face produced no warning: %+v", rep)
				}
				return
			}
			if err == nil || rep.Stage != fault.expect {
				t.Fatalf("fault %q not caught at %s (got stage=%s err=%v)",
					fault.name, fault.expect, rep.Stage, err)
			}
		})
	}
}

// TestFaultInjectionDegenerateCaughtGeometrically proves the repeated-vertex
// face (accepted with a warning by the text checker) is caught by the
// independent post-load geometric gate after the real loader builds it.
func TestFaultInjectionDegenerateCaughtGeometrically(t *testing.T) {
	path := writeSample(t, []byte("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 1 1\n"), ".obj")
	mesh, rep, err := SafeLoadOBJ(path, pt.DiffuseMaterial(pt.White), DefaultLimits())
	if err == nil {
		t.Fatalf("degenerate mesh was accepted end-to-end: tris=%d", len(mesh.Triangles))
	}
	if rep.Stage != StageDegenerate {
		t.Fatalf("expected StageDegenerate, got %s", rep.Stage)
	}
}

// TestFaultInjectionSDFNonReturning injects the three non-returning SDF
// defects and proves BoundedSDFShape always returns (bounded), never hanging.
func TestFaultInjectionSDFNonReturning(t *testing.T) {
	defects := map[string]pt.SDF{
		"nan-distance":      nanSDF{},
		"non-converging":    bigStallSDF{},
		"infinite-distance": explodingSDF{},
		"panicking":         panicSDF{},
	}
	ray := pt.Ray{
		Origin:    pt.Vector{X: -2, Y: 0, Z: 0},
		Direction: pt.Vector{X: 1, Y: 0, Z: 0},
	}
	for name, sdf := range defects {
		t.Run(name, func(t *testing.T) {
			shape := NewBoundedSDFShape(GuardTree(sdf, 64), pt.DiffuseMaterial(pt.White), DefaultLimits())
			done := make(chan pt.Hit, 1)
			go func() { done <- shape.Intersect(ray) }()
			select {
			case h := <-done:
				if h.Ok() {
					t.Fatalf("defect %s produced a false hit", name)
				}
			case <-time.After(2 * time.Second):
				t.Fatalf("defect %s caused Intersect to hang", name)
			}
		})
	}
}

// TestFaultInjectionLoaderPanicBoundary injects a panic directly into the
// guarded loader boundary and asserts it is reported, not propagated.
func TestFaultInjectionLoaderPanicBoundary(t *testing.T) {
	rep := &Report{Format: "obj"}
	_, err := guarded(func() (*pt.Mesh, error) {
		panic("runtime error: index out of range [42]")
	}, DefaultLimits(), rep)
	if err == nil || rep.Stage != StagePanic {
		t.Fatalf("injected loader panic escaped: stage=%s err=%v", rep.Stage, err)
	}
}
