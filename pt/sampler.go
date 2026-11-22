package pt

import (
	"math/rand"
)

type LightMode int

const (
	LightModeRandom = iota
	LightModeAll
)

type SpecularMode int

const (
	SpecularModeNaive = iota
	SpecularModeFirst
	SpecularModeAll
)

type BounceType int

const (
	BounceTypeAny = iota
	BounceTypeDiffuse
	BounceTypeSpecular
)

type Sampler interface {
	Sample(scene *Scene, ray Ray, rnd *rand.Rand) Color
}

func NewSampler(firstHitSamples, maxBounces int) *DefaultSampler {
	return &DefaultSampler{firstHitSamples, maxBounces, true, true, LightModeRandom, SpecularModeNaive}
}

func NewDirectSampler() *DefaultSampler {
	return &DefaultSampler{1, 0, true, false, LightModeAll, SpecularModeAll}
}

// DefaultSampler is the historical sampler facade. It keeps its original
// public fields and behavior, but its Sample method now runs through the
// composable Pipeline boundary (see transport.go): the field values are
// read on every Sample call, so mutating them between calls takes effect
// exactly as before. To customize individual stages (intersection, hit
// resolution, materials, lights, bounces, recursion), build a Pipeline
// directly or start from the one returned by Pipeline().
type DefaultSampler struct {
	FirstHitSamples int
	MaxBounces      int
	DirectLighting  bool
	SoftShadows     bool
	LightMode       LightMode
	SpecularMode    SpecularMode
}

// Pipeline expands the sampler's current configuration into a Pipeline
// using the default components. Callers may further replace individual
// stages of the returned Pipeline before using it.
func (s *DefaultSampler) Pipeline() *Pipeline {
	return &Pipeline{
		FirstHitSamples: s.FirstHitSamples,
		MaxBounces:      s.MaxBounces,
		DirectLighting:  s.DirectLighting,
		SpecularMode:    s.SpecularMode,
		Lights: DefaultLightSampler{
			SoftShadows: s.SoftShadows,
			LightMode:   s.LightMode,
		},
	}
}

func (s *DefaultSampler) Sample(scene *Scene, ray Ray, rnd *rand.Rand) Color {
	return s.Pipeline().Recurse(scene, ray, true, s.FirstHitSamples, 0, rnd)
}
