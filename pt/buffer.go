package pt

import (
	"image"
	"sync/atomic"
)

type Pixel struct {
	Samples int
	A, M, V Color
}

func (p *Pixel) Color() Color {
	if p.Samples <= 0 {
		// Defensive guard: division by zero would produce NaNs which
		// would poison the output PNG. An unsampled pixel is black.
		return Black
	}
	return p.A.DivScalar(float64(p.Samples))
}

func (p *Pixel) Variance() Color {
	if p.Samples < 2 {
		return Black
	}
	return p.V.DivScalar(float64(p.Samples - 1)).Pow(0.5)
}

func (p *Pixel) AddSample(sample Color) {
	p.Samples++
	p.A = p.A.Add(sample)
	if p.Samples == 1 {
		p.M = sample
		return
	}
	m := p.M
	p.M = p.M.Add(sample.Sub(p.M).DivScalar(float64(p.Samples)))
	p.V = p.V.Add(sample.Sub(m).Mul(sample.Sub(p.M)))
}

type Buffer struct {
	W      int
	H      int
	Pixels []Pixel
	// The counters below are maintained with atomic operations so that
	// Stats() can be called concurrently with AddSample without any
	// data race. Each individual pixel is still only touched by one
	// render worker at a time; these aggregates give readers a
	// race-free view without touching the Pixel slice.
	totalSamples   uint64
	nonEmptyPixels uint64
	maxSamples     uint64
}

func NewBuffer(w, h int) *Buffer {
	// Clamp degenerate dimensions: make() panics on negative sizes.
	// Keep the requested (possibly zero) dimensions for the stats
	// snapshot but allocate an empty, never-nil slice.
	if w < 0 {
		w = 0
	}
	if h < 0 {
		h = 0
	}
	pixels := make([]Pixel, w*h)
	return &Buffer{W: w, H: h, Pixels: pixels}
}

func (b *Buffer) Image() image.Image {
	result := image.NewRGBA64(image.Rect(0, 0, b.W, b.H))
	for y := 0; y < b.H; y++ {
		for x := 0; x < b.W; x++ {
			c := b.Pixels[y*b.W+x].Color().Pow(1 / 2.2)
			result.SetRGBA64(x, y, c.RGBA64())
		}
	}
	return result
}

func (b *Buffer) Variance(x, y int) Color {
	return b.Pixels[y*b.W+x].Variance()
}

// Samples returns the number of samples accumulated at pixel (x, y).
// It returns zero for coordinates outside the buffer instead of
// panicking, so observability callers can probe freely.
func (b *Buffer) Samples(x, y int) int {
	index := y*b.W + x
	if index < 0 || index >= len(b.Pixels) {
		return 0
	}
	return b.Pixels[index].Samples
}

func (b *Buffer) AddSample(x, y int, sample Color) {
	index := y*b.W + x
	if index < 0 || index >= len(b.Pixels) {
		// Out-of-range coordinates must never panic: pixel output is
		// the primary contract and observability must not break it.
		return
	}
	pixel := &b.Pixels[index]
	first := pixel.Samples == 0
	pixel.AddSample(sample)
	// Race-free aggregate bookkeeping.
	atomic.AddUint64(&b.totalSamples, 1)
	if first {
		atomic.AddUint64(&b.nonEmptyPixels, 1)
	}
	// Track the per-pixel maximum atomically with a CAS loop: each
	// iteration either publishes a new larger value or observes that
	// another worker already did.
	for {
		current := atomic.LoadUint64(&b.maxSamples)
		if uint64(pixel.Samples) <= current {
			break
		}
		if atomic.CompareAndSwapUint64(&b.maxSamples, current, uint64(pixel.Samples)) {
			break
		}
	}
}

// TotalSamples returns the number of samples written into the buffer
// since it was created. Monotonic and safe for concurrent reads.
func (b *Buffer) TotalSamples() uint64 {
	return atomic.LoadUint64(&b.totalSamples)
}

// Stats returns a read-only, point-in-time snapshot of the buffer. The
// returned struct holds scalar values only: it never aliases the
// internal Pixel slice, so callers cannot mutate buffer state through a
// snapshot. Safe to call concurrently with AddSample.
func (b *Buffer) Stats() BufferStats {
	stats := BufferStats{
		Width:          b.W,
		Height:         b.H,
		TotalSamples:   atomic.LoadUint64(&b.totalSamples),
		NonEmptyPixels: atomic.LoadUint64(&b.nonEmptyPixels),
		MaxSamples:     int(atomic.LoadUint64(&b.maxSamples)),
	}
	pixelCount := b.W * b.H
	if pixelCount <= 0 {
		// Zero-area buffer: all aggregates stay at their zero values
		// and MeanSamplesPerPixel must avoid a division by zero.
		return stats
	}
	stats.MeanSamplesPerPixel = float64(stats.TotalSamples) / float64(pixelCount)
	// MinSamples is derivable without scanning the Pixel slice: when
	// every pixel has at least one sample the minimum is at least one,
	// otherwise at least one pixel is still empty (minimum zero).
	if stats.NonEmptyPixels < uint64(pixelCount) {
		stats.MinSamples = 0
	} else {
		stats.MinSamples = 1
	}
	return stats
}
