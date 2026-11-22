package safety

import (
	"os"
	"path/filepath"
	"testing"
)

// The three curated samples are the permanent, reproducible regression set:
//
//	regression-cube.obj       - valid baseline (must be accepted)
//	regression-oob-index.obj  - out-of-range face index (upstream panics)
//	regression-dangling.stl   - ASCII STL with a 2-vertex dangling facet
//
// They are checked in alongside the code so `go test` reproduces the failure
// modes without depending on the random corpus being regenerated identically.

func TestRegressionCubeAccepted(t *testing.T) {
	rep, err := CheckOBJ(testdata(t, "regression-cube.obj"), DefaultLimits())
	if err != nil {
		t.Fatalf("valid cube rejected at %s: %v", rep.Stage, err)
	}
	if rep.Vertices != 8 || rep.Triangles != 12 || rep.Faces != 6 {
		t.Fatalf("unexpected counts: %+v", rep)
	}
}

func TestRegressionOOBIndexRejected(t *testing.T) {
	rep, err := CheckOBJ(testdata(t, "regression-oob-index.obj"), DefaultLimits())
	if err == nil {
		t.Fatalf("out-of-bounds face index was accepted: %+v", rep)
	}
	if rep.Stage != StageIndex {
		t.Fatalf("expected StageIndex, got %s (%v)", rep.Stage, err)
	}
	// Resource counts must reflect only what was observed before rejection.
	if rep.Vertices != 1 {
		t.Fatalf("expected 1 observed vertex, got %d", rep.Vertices)
	}
}

func TestRegressionDanglingSTLRejected(t *testing.T) {
	rep, err := CheckSTL(testdata(t, "regression-dangling.stl"), DefaultLimits())
	if err == nil {
		t.Fatalf("dangling STL facet was accepted: %+v", rep)
	}
	if rep.Stage != StageHeader {
		t.Fatalf("expected StageHeader framing error, got %s (%v)", rep.Stage, err)
	}
}

func testdata(t *testing.T, name string) string {
	t.Helper()
	p, err := filepath.Abs(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("missing regression sample %s: %v", name, err)
	}
	return p
}

// TestStageCoverage drives every documented failure stage at least once,
// proving the checker vocabulary maps onto concrete inputs rather than being
// aspirational. Each row names the stage it must produce.
func TestStageCoverage(t *testing.T) {
	limits := DefaultLimits()
	limits.MaxCoordinate = 100
	limits.MaxVertices = 4
	limits.MaxTriangles = 4
	limits.MaxFileSize = 4096

	type row struct {
		name   string
		kind   string // "obj" or "stl"
		data   []byte
		expect Stage
	}
	rows := []row{
		{"oob index", "obj", []byte("v 0 0 0\nf 1 2 3\n"), StageIndex},
		{"zero index", "obj", []byte("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n"), StageIndex},
		{"negative oob", "obj", []byte("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -9 -8 -7\n"), StageIndex},
		{"nan vertex", "obj", []byte("v nan 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"), StageValue},
		{"inf vertex", "obj", []byte("v inf 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"), StageValue},
		{"huge vertex", "obj", []byte("v 1e308 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"), StageMagnitude},
		{"truncated vertex", "obj", []byte("v 1\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"), StageHeader},
		{"two-vertex face", "obj", []byte("v 0 0 0\nv 1 0 0\nf 1 2\n"), StageHeader},
		{"noninteger index", "obj", []byte("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 x 3\n"), StageParse},
		{"no geometry", "obj", []byte("# nothing here\nv 0 0 0\n"), StageDegenerate},
		{"vertex quota", "obj", []byte("v 0 0 0\nv 0 0 0\nv 0 0 0\nv 0 0 0\nv 0 0 0\nf 1 2 3\n"), StageCount},
	}
	for _, tc := range rows {
		t.Run(tc.name, func(t *testing.T) {
			var rep *Report
			var err error
			if tc.kind == "obj" {
				rep, err = CheckOBJBytes(tc.data, limits)
			} else {
				rep, err = CheckSTLBytes(tc.data, limits)
			}
			if err == nil {
				t.Fatalf("expected rejection at %s, accepted: %+v", tc.expect, rep)
			}
			if rep.Stage != tc.expect {
				t.Fatalf("expected stage %s, got %s (err=%v)", tc.expect, rep.Stage, err)
			}
		})
	}

	// STL-specific stages.
	t.Run("stl nan float32", func(t *testing.T) {
		data := genSTLBinary(newRand(2), 1, 1, false)
		// force NaN into first vertex
		bits := mathFloat32NaN()
		data[84+12] = byte(bits)
		data[84+13] = byte(bits >> 8)
		data[84+14] = byte(bits >> 16)
		data[84+15] = byte(bits >> 24)
		rep, err := CheckSTLBytes(data, limits)
		if err == nil || rep.Stage != StageValue {
			t.Fatalf("expected StageValue, got %s err=%v", rep.Stage, err)
		}
	})
	t.Run("stl corrupt header", func(t *testing.T) {
		data := genSTLBinary(newRand(3), 1, 999, true)
		rep, err := CheckSTLBytes(data, limits)
		if err == nil || rep.Stage != StageHeader {
			t.Fatalf("expected StageHeader, got %s err=%v", rep.Stage, err)
		}
	})
	t.Run("stl dangling ascii", func(t *testing.T) {
		data := genSTLAscii(newRand(4), 1, true)
		rep, err := CheckSTLBytes(data, limits)
		if err == nil || rep.Stage != StageHeader {
			t.Fatalf("expected StageHeader, got %s err=%v", rep.Stage, err)
		}
	})
	t.Run("oversized sample", func(t *testing.T) {
		small := DefaultLimits()
		small.MaxFileSize = 16
		rep, err := CheckOBJBytes([]byte("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"), small)
		if err == nil || rep.Stage != StageSize {
			t.Fatalf("expected StageSize, got %s err=%v", rep.Stage, err)
		}
	})
}
