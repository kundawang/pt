package pt

import (
	"math"
	"math/rand"
)

// This file defines the light-transport boundary of the renderer. The
// monolithic DefaultSampler is decomposed into narrow, individually
// replaceable interfaces, and a Pipeline that wires them together.
//
// The interfaces are:
//
//	Intersector        ray -> Hit                 (geometry query)
//	HitResolver        Hit + Ray -> HitInfo       (position/normal/inside)
//	MaterialEvaluator  Shape + point -> Material  (textures applied)
//	EnvironmentSampler miss shading
//	LightSampler       direct lighting
//	Bouncer            next path vertex
//	Recurser           recursive path evaluation
//
// All default implementations are stateless value types: they keep no
// mutable scratch state between calls and are safe for concurrent use by
// multiple goroutines. All randomness flows through the *rand.Rand
// argument; implementations must not touch the global rand source unless
// rnd is nil (see randomIntn and RandomUnitVector).
//
// ---------------------------------------------------------------------------
// Call-order and RNG-consumption contract of the default Pipeline
// ---------------------------------------------------------------------------
//
// Pipeline.Recurse(scene, ray, emission, samples, depth, rnd) performs, in
// exactly this order:
//
//  1. Recursion cap: if depth > MaxBounces, return Black. No randoms drawn.
//  2. Intersector.Intersect(ray). No randoms drawn. (The default intersector
//     is the scene itself, preserving Scene.RayCount.)
//  3. Miss handling: if the hit is not Ok(), or carries neither a Shape nor
//     a cached HitInfo, return EnvironmentSampler.SampleEnvironment.
//     No randoms drawn.
//  4. HitResolver.Info(hit, ray): computes position, normal (flipped when
//     the ray hits from behind, except for Volume/SDFShape), the inside
//     flag, and resolves the material through MaterialEvaluator.
//     MaterialEvaluator applies, in order: Shape.MaterialAt, then
//     Material.Texture -> Color, then Material.GlossTexture -> Gloss.
//     (NormalTexture/BumpTexture are consulted inside Shape.NormalAt for
//     triangles, before the material textures.) No randoms drawn.
//  5. Emission: if material.Emittance > 0 and DirectLighting is on and
//     emission is false, return Black immediately. Otherwise accumulate
//     material.Color * Emittance * samples. No randoms drawn.
//  6. Stratification: n = sqrt(samples). The bounce-type range is
//     [Diffuse, Specular] when SpecularMode == SpecularModeAll, or depth == 0
//     and SpecularMode == SpecularModeFirst; otherwise [Any, Any].
//  7. For each (u, v, mode) in row-major u -> v -> mode order: draw
//     fu = (u + rnd)/n then fv = (v + rnd)/n (2 draws), then
//     Bouncer.Bounce. The default bouncer draws 1 random for the
//     reflect/refract test under BounceTypeAny (0 under Diffuse/Specular),
//     then ConeBounce/WeightedBounce draw via RandomUnitVector (3 draws per
//     rejection-sampling attempt; ConeBounce with Gloss < EPS draws none).
//     When mode == BounceTypeAny the probability is forced to 1.
//  8. Reflected rays recurse through Recurser.Recurse at depth+1 with
//     samples=1 and emission=reflected, then the result is tinted by
//     material.Color and material.Tint. Non-reflected (diffuse) rays
//     recurse first, then — when DirectLighting is on —
//     LightSampler.SampleLights is evaluated at the shading point and added
//     to the indirect term before multiplying by material.Color.
//  9. LightSampler: LightModeAll evaluates every light in scene.Lights
//     order; LightModeRandom draws 1 random (rnd.Intn) to pick one light
//     and scales by len(lights). Per light, SoftShadows rejection-samples a
//     disk point (2 draws per attempt) plus one RandomUnitVector; without
//     SoftShadows no randoms are drawn. A light contributes only if the
//     cosine term is positive and a shadow ray hits that light first.
//
// Error-handling contract:
//
//   - No hit, depth beyond MaxBounces, and a Bouncer that never terminates
//     a path ("cyclic" materials) are all bounded: they return Black or the
//     environment color without panicking.
//   - A Hit with a nil Shape and no cached HitInfo is treated as a miss.
//   - A nil Shape passed to the default MaterialEvaluator yields the zero
//     Material instead of panicking.
//   - Nil Pipeline components fall back to the default implementations.
//   - When RecoverPanics is set, a panic inside any component is recovered
//     and that (sub-)path returns Black.

// Intersector reports the closest intersection of a ray with the scene.
// *Scene implements it (preserving the atomic ray counter).
type Intersector interface {
	Intersect(r Ray) Hit
}

// HitResolver expands a Hit into full shading information: position,
// normal, inside flag and the (textured) material.
type HitResolver interface {
	Info(hit Hit, ray Ray) HitInfo
}

// MaterialEvaluator resolves the material of a shape at a point, applying
// any textures. It is the replaceable seam behind the package-level
// MaterialAt function.
type MaterialEvaluator interface {
	MaterialAt(shape Shape, point Vector) Material
}

// EnvironmentSampler shades a ray that missed all geometry.
type EnvironmentSampler interface {
	SampleEnvironment(scene *Scene, ray Ray) Color
}

// LightSampler computes direct lighting at a shading point. n is the
// shading "ray": Origin is the surface point, Direction the normal.
type LightSampler interface {
	SampleLights(scene *Scene, n Ray, rnd *rand.Rand) Color
}

// Bouncer generates the next ray of a path at a shading point, returning
// the new ray, whether it was a specular (reflection/refraction) bounce,
// and the probability weight p of that bounce.
type Bouncer interface {
	Bounce(in Ray, info *HitInfo, u, v float64, bounceType BounceType, rnd *rand.Rand) (out Ray, reflected bool, p float64)
}

// Recurser evaluates a bounced ray. It is the recursion seam of the
// transport loop; Pipeline itself implements it.
type Recurser interface {
	Recurse(scene *Scene, ray Ray, emission bool, samples, depth int, rnd *rand.Rand) Color
}

// DefaultMaterialEvaluator is the stock MaterialEvaluator. It applies the
// color texture first, then the gloss texture, matching the historical
// MaterialAt behavior.
type DefaultMaterialEvaluator struct{}

func (DefaultMaterialEvaluator) MaterialAt(shape Shape, point Vector) Material {
	if shape == nil {
		return Material{}
	}
	material := shape.MaterialAt(point)
	uv := shape.UV(point)
	if material.Texture != nil {
		material.Color = material.Texture.Sample(uv.X, uv.Y)
	}
	if material.GlossTexture != nil {
		c := material.GlossTexture.Sample(uv.X, uv.Y)
		material.Gloss = (c.R + c.G + c.B) / 3
	}
	return material
}

// DefaultHitResolver is the stock HitResolver. Materials may be nil, in
// which case DefaultMaterialEvaluator is used.
type DefaultHitResolver struct {
	Materials MaterialEvaluator
}

func (r DefaultHitResolver) evaluator() MaterialEvaluator {
	if r.Materials != nil {
		return r.Materials
	}
	return DefaultMaterialEvaluator{}
}

func (r DefaultHitResolver) Info(hit Hit, ray Ray) HitInfo {
	if hit.HitInfo != nil {
		return *hit.HitInfo
	}
	shape := hit.Shape
	if shape == nil {
		// Degenerate hit: no geometry to shade. Return a zero HitInfo
		// rather than dereferencing a nil Shape.
		return HitInfo{}
	}
	position := ray.Position(hit.T)
	normal := shape.NormalAt(position)
	material := r.evaluator().MaterialAt(shape, position)
	inside := false
	if normal.Dot(ray.Direction) > 0 {
		normal = normal.Negate()
		inside = true
		switch shape.(type) {
		case *Volume, *SDFShape:
			inside = false
		}
	}
	return HitInfo{shape, position, normal, Ray{position, normal}, material, inside}
}

// DefaultEnvironmentSampler is the stock EnvironmentSampler: the scene
// texture if present, otherwise the flat scene color.
type DefaultEnvironmentSampler struct{}

func (DefaultEnvironmentSampler) SampleEnvironment(scene *Scene, ray Ray) Color {
	if scene.Texture != nil {
		d := ray.Direction
		u := math.Atan2(d.Z, d.X) + scene.TextureAngle
		v := math.Atan2(d.Y, Vector{d.X, 0, d.Z}.Length())
		u = (u + math.Pi) / (2 * math.Pi)
		v = (v + math.Pi/2) / math.Pi
		return scene.Texture.Sample(u, v)
	}
	return scene.Color
}

// DefaultLightSampler is the stock LightSampler. It is a stateless value;
// SoftShadows and LightMode are read on every call.
type DefaultLightSampler struct {
	SoftShadows bool
	LightMode   LightMode
}

func (s DefaultLightSampler) SampleLights(scene *Scene, n Ray, rnd *rand.Rand) Color {
	nLights := len(scene.Lights)
	if nLights == 0 {
		return Black
	}
	if s.LightMode == LightModeAll {
		var result Color
		for _, light := range scene.Lights {
			result = result.Add(s.sampleLight(scene, n, rnd, light))
		}
		return result
	}
	// pick a random light (through rnd, never the global source)
	light := scene.Lights[randomIntn(rnd, nLights)]
	return s.sampleLight(scene, n, rnd, light).MulScalar(float64(nLights))
}

func (s DefaultLightSampler) sampleLight(scene *Scene, n Ray, rnd *rand.Rand, light Shape) Color {
	// get bounding sphere center and radius
	var center Vector
	var radius float64
	switch t := light.(type) {
	case *Sphere:
		radius = t.Radius
		center = t.Center
	default:
		// get bounding sphere from bounding box
		box := t.BoundingBox()
		radius = box.OuterRadius()
		center = box.Center()
	}

	// get random point in disk
	point := center
	if s.SoftShadows {
		for {
			x := rnd.Float64()*2 - 1
			y := rnd.Float64()*2 - 1
			if x*x+y*y <= 1 {
				l := center.Sub(n.Origin).Normalize()
				u := l.Cross(RandomUnitVector(rnd)).Normalize()
				v := l.Cross(u)
				point = Vector{}
				point = point.Add(u.MulScalar(x * radius))
				point = point.Add(v.MulScalar(y * radius))
				point = point.Add(center)
				break
			}
		}
	}

	// construct ray toward light point
	ray := Ray{n.Origin, point.Sub(n.Origin).Normalize()}

	// get cosine term
	diffuse := ray.Direction.Dot(n.Direction)
	if diffuse <= 0 {
		return Black
	}

	// check for light visibility
	hit := scene.Intersect(ray)
	if !hit.Ok() || hit.Shape != light {
		return Black
	}

	// compute solid angle (hemisphere coverage)
	hyp := center.Sub(n.Origin).Length()
	opp := radius
	theta := math.Asin(opp / hyp)
	adj := opp / math.Tan(theta)
	d := math.Cos(theta) * adj
	r := math.Sin(theta) * adj
	coverage := (r * r) / (d * d)

	// TODO: fix issue where hyp < opp (point inside sphere)
	if hyp < opp {
		coverage = 1
	}
	coverage = math.Min(coverage, 1)

	// get material properties from light
	material := MaterialAt(light, point)

	// combine factors
	m := material.Emittance * diffuse * coverage
	return material.Color.MulScalar(m)
}

// DefaultBouncer is the stock Bouncer; it defers to Ray.Bounce, preserving
// the historical reflection/refraction/diffuse logic and RNG consumption.
type DefaultBouncer struct{}

func (DefaultBouncer) Bounce(in Ray, info *HitInfo, u, v float64, bounceType BounceType, rnd *rand.Rand) (Ray, bool, float64) {
	return in.Bounce(info, u, v, bounceType, rnd)
}

// Pipeline is the composable light-transport engine. Each stage is a
// narrow interface; a nil stage falls back to the default implementation.
// Pipeline implements both Sampler (Sample) and Recurser (Recurse), so it
// can be passed to Render directly or nested inside another Pipeline.
//
// The zero Pipeline is valid and equivalent to a DefaultSampler with zero
// fields (plus LightModeRandom / no soft shadows from the zero
// DefaultLightSampler).
type Pipeline struct {
	FirstHitSamples int
	MaxBounces      int
	DirectLighting  bool
	SpecularMode    SpecularMode

	Intersector Intersector        // default: the scene itself
	Resolver    HitResolver        // default: DefaultHitResolver
	Environment EnvironmentSampler // default: DefaultEnvironmentSampler
	Lights      LightSampler       // default: zero DefaultLightSampler
	Bouncer     Bouncer            // default: DefaultBouncer
	Recurser    Recurser           // default: this pipeline itself

	// RecoverPanics, when true, recovers panics raised by any component
	// and shades the affected (sub-)path Black instead of crashing.
	RecoverPanics bool
}

// Sample implements Sampler: it traces a camera ray with FirstHitSamples
// stratified first-hit samples.
func (p *Pipeline) Sample(scene *Scene, ray Ray, rnd *rand.Rand) Color {
	return p.Recurse(scene, ray, true, p.FirstHitSamples, 0, rnd)
}

// Recurse implements Recurser.
func (p *Pipeline) Recurse(scene *Scene, ray Ray, emission bool, samples, depth int, rnd *rand.Rand) Color {
	if p.RecoverPanics {
		return p.recurseSafe(scene, ray, emission, samples, depth, rnd)
	}
	return p.recurse(scene, ray, emission, samples, depth, rnd)
}

func (p *Pipeline) recurseSafe(scene *Scene, ray Ray, emission bool, samples, depth int, rnd *rand.Rand) (result Color) {
	defer func() {
		if recover() != nil {
			result = Black
		}
	}()
	return p.recurse(scene, ray, emission, samples, depth, rnd)
}

func (p *Pipeline) intersector(scene *Scene) Intersector {
	if p.Intersector != nil {
		return p.Intersector
	}
	return scene
}

func (p *Pipeline) resolver() HitResolver {
	if p.Resolver != nil {
		return p.Resolver
	}
	return DefaultHitResolver{}
}

func (p *Pipeline) environment() EnvironmentSampler {
	if p.Environment != nil {
		return p.Environment
	}
	return DefaultEnvironmentSampler{}
}

func (p *Pipeline) lights() LightSampler {
	if p.Lights != nil {
		return p.Lights
	}
	return DefaultLightSampler{}
}

func (p *Pipeline) bouncer() Bouncer {
	if p.Bouncer != nil {
		return p.Bouncer
	}
	return DefaultBouncer{}
}

func (p *Pipeline) recurser() Recurser {
	if p.Recurser != nil {
		return p.Recurser
	}
	return p
}

func (p *Pipeline) recurse(scene *Scene, ray Ray, emission bool, samples, depth int, rnd *rand.Rand) Color {
	if depth > p.MaxBounces {
		return Black
	}
	hit := p.intersector(scene).Intersect(ray)
	if !hit.Ok() || (hit.Shape == nil && hit.HitInfo == nil) {
		return p.environment().SampleEnvironment(scene, ray)
	}
	info := p.resolver().Info(hit, ray)
	material := info.Material
	result := Black
	if material.Emittance > 0 {
		if p.DirectLighting && !emission {
			return Black
		}
		result = result.Add(material.Color.MulScalar(material.Emittance * float64(samples)))
	}
	n := int(math.Sqrt(float64(samples)))
	var ma, mb BounceType
	if p.SpecularMode == SpecularModeAll || (depth == 0 && p.SpecularMode == SpecularModeFirst) {
		ma = BounceTypeDiffuse
		mb = BounceTypeSpecular
	} else {
		ma = BounceTypeAny
		mb = BounceTypeAny
	}
	bouncer := p.bouncer()
	recurser := p.recurser()
	for u := 0; u < n; u++ {
		for v := 0; v < n; v++ {
			for mode := ma; mode <= mb; mode++ {
				fu := (float64(u) + rnd.Float64()) / float64(n)
				fv := (float64(v) + rnd.Float64()) / float64(n)
				newRay, reflected, prob := bouncer.Bounce(ray, &info, fu, fv, mode, rnd)
				if mode == BounceTypeAny {
					prob = 1
				}
				if prob > 0 && reflected {
					// specular
					indirect := recurser.Recurse(scene, newRay, reflected, 1, depth+1, rnd)
					tinted := indirect.Mix(material.Color.Mul(indirect), material.Tint)
					result = result.Add(tinted.MulScalar(prob))
				}
				if prob > 0 && !reflected {
					// diffuse
					indirect := recurser.Recurse(scene, newRay, reflected, 1, depth+1, rnd)
					direct := Black
					if p.DirectLighting {
						direct = p.lights().SampleLights(scene, info.Ray, rnd)
					}
					result = result.Add(material.Color.Mul(direct.Add(indirect)).MulScalar(prob))
				}
			}
		}
	}
	return result.DivScalar(float64(n * n))
}

// randomIntn draws from rnd when available, falling back to the global
// source (matching RandomUnitVector's nil-rnd behavior).
func randomIntn(rnd *rand.Rand, n int) int {
	if rnd == nil {
		return rand.Intn(n)
	}
	return rnd.Intn(n)
}
