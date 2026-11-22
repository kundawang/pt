// Package-level note on the direction-number table in sobol_table.go:
//
// The Sobol direction numbers are from the published Joe & Kuo (2008) table
// "new-joe-kuo-6.21201" (dimensions 2..1024). That data set is distributed
// under a BSD-style licence by Frances Y. Kuo and Stephen Joe; the following
// notice is retained as required:
//
//	Copyright (c) 2008, Frances Y. Kuo and Stephen Joe
//	All rights reserved.
//	Redistribution and use in source and binary forms, with or without
//	modification, are permitted provided that the copyright notice, this
//	list of conditions and the disclaimer are retained. Neither the names of
//	the copyright holders nor the names of the University of New South Wales
//	and the University of Waikato may be used to endorse or promote products
//	derived from this software without specific prior written permission.
//	THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.

package pt

import (
	"errors"
	"math"
	"math/bits"
	"sync"
)

// This file implements deterministic low-discrepancy sampling for the path
// tracer. Every value is a pure function of (pixel, sample, dimension, seed):
//
//     v = Sobol.Value(pixel, sample, dimension)
//
// The underlying point set is the Sobol sequence from the published Joe & Kuo
// direction numbers (see sobol_table.go, dimensions 0..1023). The sequence is
// indexed in canonical Gray-code order, so prefix sample counts match the
// reference implementation. Each dimension is scrambled with a per-pixel,
// per-dimension digital shift (XOR) derived from the seed with SplitMix64.
// A digital shift maps elementary intervals onto elementary intervals, so the
// (t,m,s)-net stratification of the Sobol points is preserved.
//
// Because Value is pure and Sobol is immutable, concurrent reads are safe and
// no mutable RNG is shared between pixels, samples, workers or tiles.
//
// Dimension exhaustion: dimensions 0..1023 are true Sobol dimensions. A path
// that consumes more than 1024 values (possible with deep bounces) continues
// with a deterministic SplitMix64 hash of (seed, pixel, sample, dimension)
// instead of panicking. The low-discrepancy guarantee therefore applies to
// the first 1024 dimensions; beyond that values are deterministic and
// effectively independent, but no longer low-discrepancy. A single path
// rarely reaches that many dimensions.
//
// Negative pixel or sample indices are clamped to 0. A nil Sobol source is
// treated as seed 0. Sequences may be jumped to directly with Value, or with
// SobolRand.SeekTo/Skip/SkipSamples; a negative Skip is a no-op.

const (
	// sobolMaxDims is the number of available Sobol dimensions
	// (dimension 0 is the van der Corput sequence plus 1023 table entries).
	sobolMaxDims = 1024
	sobolBits    = 64
)

const sobolInv2p53 = 1.0 / 9007199254740992.0 // 1 / 2^53

// sobolParam and sobolParams are generated from the Joe & Kuo direction
// number table; see sobol_table.go.

var (
	sobolDirsOnce sync.Once
	sobolDirs     [sobolMaxDims][sobolBits]uint64
)

// buildSobolDirections expands the compact (s, a, m) description of every
// dimension into the 64 direction numbers used by the sequence.
func buildSobolDirections() {
	// Dimension 0 is the van der Corput sequence (all initial m's are 1).
	for k := 0; k < sobolBits; k++ {
		sobolDirs[0][k] = 1 << uint(sobolBits-1-k)
	}
	for d := 1; d < sobolMaxDims; d++ {
		p := sobolParams[d-1]
		s := int(p.s)
		a := p.a
		for k := 0; k < sobolBits; k++ {
			if k < s {
				sobolDirs[d][k] = p.m[k] << uint(sobolBits-1-k)
				continue
			}
			// V[k] = V[k-s] ^ (V[k-s] >> s) ^ sum(a_r * V[k-r])
			v := sobolDirs[d][k-s] ^ (sobolDirs[d][k-s] >> uint(s))
			for r := 1; r <= s-1; r++ {
				if (a>>uint(s-1-r))&1 != 0 {
					v ^= sobolDirs[d][k-r]
				}
			}
			sobolDirs[d][k] = v
		}
	}
}

func sobolDirectionTable() *[sobolMaxDims][sobolBits]uint64 {
	sobolDirsOnce.Do(buildSobolDirections)
	return &sobolDirs
}

// sobolRawPoint returns the unscrambled Sobol point for the given sequence
// index in dimension d. index is interpreted in canonical Gray-code order.
func sobolRawPoint(index uint64, d int) uint64 {
	dirs := sobolDirectionTable()
	g := index ^ (index >> 1)
	var x uint64
	for g != 0 {
		k := bits.TrailingZeros64(g)
		x ^= dirs[d][k]
		g &= g - 1
	}
	return x
}

// ErrNonFiniteSeed is returned by NewSobolFromSeed for NaN or infinite seeds.
var ErrNonFiniteSeed = errors.New("pt: Sobol seed must be finite")

// Sobol is an immutable deterministic sample source. Construct it once and
// share it between goroutines; Value is safe for concurrent use.
type Sobol struct {
	seed uint64
}

// NewSobol returns a Sobol source with the given seed. Any uint64 value is
// valid, including 0. Different seeds produce different digital shifts and
// therefore different (but equally well-stratified) sample sets.
func NewSobol(seed uint64) *Sobol {
	sobolDirectionTable()
	return &Sobol{seed: seed}
}

// NewSobolFromSeed is a convenience constructor that validates a seed given
// as a float64. NaN and ±Inf are rejected with ErrNonFiniteSeed; finite seeds
// (including -0) are converted with their IEEE-754 bit pattern.
func NewSobolFromSeed(seed float64) (*Sobol, error) {
	if math.IsNaN(seed) || math.IsInf(seed, 0) {
		return nil, ErrNonFiniteSeed
	}
	return NewSobol(math.Float64bits(seed)), nil
}

// Seed returns the seed used by the source.
func (s *Sobol) Seed() uint64 {
	return s.seed
}

func splitmix64(x uint64) uint64 {
	x += 0x9E3779B97F4A7C15
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
	x = (x ^ (x >> 27)) * 0x94D049BB133111EB
	return x ^ (x >> 31)
}

// scramble returns the 64-bit digital shift for one pixel and Sobol
// dimension. It is constant for a pixel and dimension across all samples,
// which is what makes each pixel's set of samples a shifted Sobol point set.
func (s *Sobol) scramble(pixel int64, dim int) uint64 {
	h := splitmix64(s.seed ^ 0xA0761D6478BD642F)
	h = splitmix64(h ^ uint64(pixel))
	h = splitmix64(h ^ uint64(dim))
	return h
}

// exhaustedValue is the documented deterministic fallback for dimensions
// beyond the 1024 available Sobol dimensions. It is a pure hash of all four
// indices, so it stays reproducible without sharing state.
func (s *Sobol) exhaustedValue(pixel, sample int64, dim uint64) uint64 {
	h := splitmix64(s.seed ^ 0xE7037ED1A0B428DB)
	h = splitmix64(h ^ uint64(pixel))
	h = splitmix64(h ^ uint64(sample))
	h = splitmix64(h ^ dim*0x9E3779B97F4A7C15)
	return h
}

// Value returns the sample value in [0, 1) for the given pixel, sample
// ordinal and dimension. It is a pure function: the result depends only on
// its arguments and the source's seed, never on call order, scheduling,
// worker count or tiling.
//
// Negative pixel and sample indices are clamped to 0. Dimensions beyond
// sobolMaxDims use the deterministic fallback described in the file comment.
func (s *Sobol) Value(pixel, sample int64, dim uint64) float64 {
	if pixel < 0 {
		pixel = 0
	}
	if sample < 0 {
		sample = 0
	}
	var x uint64
	if dim < sobolMaxDims {
		x = sobolRawPoint(uint64(sample), int(dim)) ^ s.scramble(pixel, int(dim))
	} else {
		x = s.exhaustedValue(pixel, sample, dim)
	}
	return float64(x>>11) * sobolInv2p53
}

// TraceSamples returns the first n values a path would consume for the given
// pixel and sample ordinal, one per dimension. It exists so that the exact
// sampling order can be audited, logged or compared independently of
// rendering and thread scheduling.
func (s *Sobol) TraceSamples(pixel, sample int64, n int) []float64 {
	if n <= 0 {
		return nil
	}
	result := make([]float64, n)
	for i := range result {
		result[i] = s.Value(pixel, sample, uint64(i))
	}
	return result
}

// SobolRand is a per-path cursor over a Sobol source. It implements Random so
// that it can be threaded through the camera, light sampling and materials
// exactly like math/rand.Rand. Each path/sample must own its own SobolRand;
// it is not safe for concurrent use, and it is never shared.
type SobolRand struct {
	seq           *Sobol
	pixel, sample int64
	dim           uint64
}

// NewSobolRand returns a cursor positioned at dimension 0 of the given
// pixel/sample pair. A nil seq selects NewSobol(0). Negative indices clamp
// to 0.
func NewSobolRand(seq *Sobol, pixel, sample int64) *SobolRand {
	if seq == nil {
		seq = NewSobol(0)
	}
	r := &SobolRand{seq: seq}
	r.SeekTo(pixel, sample, 0)
	return r
}

// SeekTo jumps the cursor to an arbitrary (pixel, sample, dimension) position.
// Jumping is O(64) regardless of how far it moves, so no iteration through
// discarded samples is required. Negative pixel/sample clamp to 0.
func (r *SobolRand) SeekTo(pixel, sample int64, dim uint64) {
	if pixel < 0 {
		pixel = 0
	}
	if sample < 0 {
		sample = 0
	}
	r.pixel = pixel
	r.sample = sample
	r.dim = dim
}

// Reset restarts the cursor at dimension 0 of a new sample.
func (r *SobolRand) Reset(pixel, sample int64) {
	r.SeekTo(pixel, sample, 0)
}

// Skip advances the cursor by dims dimensions. A non-positive count is a
// no-op with a defined result.
func (r *SobolRand) Skip(dims int) {
	if dims > 0 {
		r.dim += uint64(dims)
	}
}

// SkipSamples jumps forward by samples sample ordinals without iterating
// through the skipped points. This is the O(1) "sequence jump" used, for
// example, to give each render iteration a disjoint sample set. A
// non-positive count is a no-op. Negative resulting ordinals clamp to 0.
func (r *SobolRand) SkipSamples(samples int64) {
	if samples > 0 {
		r.sample += samples
	}
}

// NextSample advances to a fresh Sobol point (the next sample ordinal) and
// resets the dimension cursor to 0. Callers that drive a path explicitly use
// it to begin each path at an independent, well-stratified point instead of
// continuing through an ever-growing dimension index. The renderer itself
// seeks directly with SeekTo, which is equivalent and O(1).
func (r *SobolRand) NextSample() {
	r.sample++
	if r.sample < 0 {
		r.sample = 0
	}
	r.dim = 0
}

// Dimension returns the next dimension index to be consumed.
func (r *SobolRand) Dimension() uint64 {
	return r.dim
}

// SampleIndex returns the sample ordinal the cursor is positioned at.
func (r *SobolRand) SampleIndex() int64 {
	return r.sample
}

// Float64 returns the next sample value in [0, 1) and advances one dimension.
// Dimensions beyond sobolMaxDims use Value's documented deterministic
// fallback.
func (r *SobolRand) Float64() float64 {
	v := r.seq.Value(r.pixel, r.sample, r.dim)
	r.dim++
	return v
}

// Intn returns a deterministic integer in [0, n). It consumes one dimension.
// For n <= 0 it returns 0 instead of panicking, so a degenerate scene cannot
// abort a render.
func (r *SobolRand) Intn(n int) int {
	if n <= 0 {
		return 0
	}
	v := int(r.Float64() * float64(n))
	if v >= n {
		v = n - 1
	}
	return v
}

// SobolSeedForIteration derives the seed used for one render iteration.
// Iteration 0 uses seed unchanged; later iterations use a SplitMix64 domain
// separation so each iteration gets an independent but reproducible Sobol
// point set. Iterative renderers call this instead of letting the sample
// ordinal grow without bound, which keeps sample indices small and preserves
// the low-discrepancy guarantee for long progressive renders.
func SobolSeedForIteration(seed uint64, iteration int) uint64 {
	if iteration <= 0 {
		return seed
	}
	return splitmix64(seed ^ (uint64(iteration) * 0x9E3779B97F4A7C15))
}

// SobolSeedForTile derives a seed for a tile rectangle. Tiled renders that
// want per-tile independence can use it; the deterministic renderer does not
// need it because Value already keys on absolute pixel index, making tiles
// bit-identical to a whole-image render. w and h are reserved for future
// size-dependent decorrelation and do not currently affect the result.
func SobolSeedForTile(seed uint64, x, y, w, h int) uint64 {
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	key := uint64(y)<<32 | uint64(x)
	return splitmix64(seed ^ splitmix64(key+0x9E3779B97F4A7C15))
}
