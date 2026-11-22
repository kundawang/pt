package safety

import (
	"math"

	"github.com/fogleman/pt/pt"
)

// CheckMesh is the post-load geometric sanity pass. It runs after a sample
// has passed the independent reference check and been parsed by the real
// loader, proving the two layers agree. It catches defects the text-level
// checker cannot see: a loader that silently drops or reorders vertices,
// non-finite coordinates introduced by float conversion, and zero-area
// triangles that would poison k-d tree splitting.
func CheckMesh(mesh *pt.Mesh, report *Report, limits Limits) (*Report, error) {
	limits = limits.fill()
	if mesh == nil {
		return report.fail(StagePanic, rejected(StagePanic, "loader returned a nil mesh"))
	}
	triangles := mesh.Triangles
	if len(triangles) == 0 {
		return report.fail(StageDegenerate, rejected(StageDegenerate,
			"loader produced zero triangles (reference saw %d)", report.Triangles))
	}
	if len(triangles) > limits.MaxTriangles {
		return report.fail(StageCount, rejected(StageCount,
			"loader produced %d triangles, limit %d", len(triangles), limits.MaxTriangles))
	}
	degenerate := 0
	for i, t := range triangles {
		for _, v := range [3]pt.Vector{t.V1, t.V2, t.V3} {
			if !finite(v.X) || !finite(v.Y) || !finite(v.Z) {
				return report.fail(StageValue, rejected(StageValue,
					"triangle %d contains a non-finite vertex after load", i))
			}
			if !bounded(v.X, limits.MaxCoordinate) || !bounded(v.Y, limits.MaxCoordinate) || !bounded(v.Z, limits.MaxCoordinate) {
				return report.fail(StageMagnitude, rejected(StageMagnitude,
					"triangle %d vertex exceeds coordinate limit %g after load", i, limits.MaxCoordinate))
			}
		}
		// Cross-product area; a collapsed triangle yields ~zero.
		e1 := t.V2.Sub(t.V1)
		e2 := t.V3.Sub(t.V1)
		cx := e1.Y*e2.Z - e1.Z*e2.Y
		cy := e1.Z*e2.X - e1.X*e2.Z
		cz := e1.X*e2.Y - e1.Y*e2.X
		area := 0.5 * math.Sqrt(cx*cx+cy*cy+cz*cz)
		if math.IsNaN(area) || area < degenerateArea {
			degenerate++
			report.warn(sprintf("triangle %d is degenerate (area=%g)", i, area))
		}
	}
	if degenerate == len(triangles) {
		return report.fail(StageDegenerate, rejected(StageDegenerate,
			"all %d triangles are degenerate", degenerate))
	}
	report.Vertices = countDistinctVertices(triangles)
	report.Triangles = len(triangles)
	report.Stage = StageAccepted
	return report, nil
}

// countDistinctVertices counts unique position vectors, matching the memory a
// mesh effectively occupies for k-d construction.
func countDistinctVertices(triangles []*pt.Triangle) int {
	seen := make(map[pt.Vector]struct{}, len(triangles)*2)
	for _, t := range triangles {
		seen[t.V1] = struct{}{}
		seen[t.V2] = struct{}{}
		seen[t.V3] = struct{}{}
	}
	return len(seen)
}
