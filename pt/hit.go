package pt

type Hit struct {
	Shape   Shape
	T       float64
	HitInfo *HitInfo
}

type HitInfo struct {
	Shape    Shape
	Position Vector
	Normal   Vector
	Ray      Ray
	Material Material
	Inside   bool
}

var NoHit = Hit{nil, INF, nil}

func (hit *Hit) Ok() bool {
	return hit.T < INF
}

// Info expands the hit into full shading information using the default
// HitResolver (see transport.go). A nil Shape yields a zero HitInfo
// instead of panicking.
func (hit *Hit) Info(r Ray) HitInfo {
	return DefaultHitResolver{}.Info(*hit, r)
}
