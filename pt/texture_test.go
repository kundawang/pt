package pt

import (
	"image"
	"image/color"
	"math"
	"math/rand"
	"testing"
)

func checkerImage(n, cell int) *image.RGBA {
	im := image.NewRGBA(image.Rect(0, 0, n, n))
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			v := uint8(255)
			if (x/cell+y/cell)%2 == 0 {
				v = 0
			}
			im.SetRGBA(x, y, color.RGBA{v, v, v, 255})
		}
	}
	return im
}

func gradientImage(w, h int) *image.RGBA {
	im := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := uint8(x * 255 / (w - 1))
			im.SetRGBA(x, y, color.RGBA{v, v, v, 255})
		}
	}
	return im
}

func noiseImage(w, h int, seed int64) *image.RGBA {
	rnd := rand.New(rand.NewSource(seed))
	im := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := uint8(rnd.Intn(256))
			im.SetRGBA(x, y, color.RGBA{v, uint8(rnd.Intn(256)), uint8(rnd.Intn(256)), 255})
		}
	}
	return im
}

func approxColor(a, b Color, eps float64) bool {
	return math.Abs(a.R-b.R) < eps && math.Abs(a.G-b.G) < eps && math.Abs(a.B-b.B) < eps
}

func meanColor(data []Color) Color {
	c := Black
	for _, d := range data {
		c = c.Add(d)
	}
	return c.DivScalar(float64(len(data)))
}

func finiteColor(c Color) bool {
	f := func(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }
	return f(c.R) && f(c.G) && f(c.B)
}

// TestSampleMatchesOriginal verifies the refactored Sample is bit-identical
// to the original bilinear implementation for ordinary and wrapped UVs.
func TestSampleMatchesOriginal(t *testing.T) {
	tex := NewTexture(noiseImage(37, 23, 1)).(*ColorTexture)
	ref := func(u, v float64) Color {
		u = Fract(Fract(u) + 1)
		v = Fract(Fract(v) + 1)
		v = 1 - v
		w := float64(tex.Width) - 1
		h := float64(tex.Height) - 1
		X, x := math.Modf(u * w)
		Y, y := math.Modf(v * h)
		x0 := int(X)
		y0 := int(Y)
		x1 := x0 + 1
		y1 := y0 + 1
		c00 := tex.Data[y0*tex.Width+x0]
		c01 := tex.Data[y1*tex.Width+x0]
		c10 := tex.Data[y0*tex.Width+x1]
		c11 := tex.Data[y1*tex.Width+x1]
		c := Black
		c = c.Add(c00.MulScalar((1 - x) * (1 - y)))
		c = c.Add(c10.MulScalar(x * (1 - y)))
		c = c.Add(c01.MulScalar((1 - x) * y))
		c = c.Add(c11.MulScalar(x * y))
		return c
	}
	// The grid avoids exact integers, where the original code indexed out
	// of range (v wraps to 0, the 1-v flip yields exactly 1).
	for i := 0; i < 2000; i++ {
		u := -2.7 + 5.3*float64(i)/1999
		v := -2.9 + 5.7*float64((i*7919)%2000)/1999
		got := tex.Sample(u, v)
		want := ref(u, v)
		if got != want {
			t.Fatalf("Sample(%v, %v) = %v, want %v", u, v, got, want)
		}
	}
}

// TestSampleUnchangedByMipmaps verifies building a mip chain never changes
// the plain Sample results.
func TestSampleUnchangedByMipmaps(t *testing.T) {
	im := noiseImage(64, 48, 2)
	plain := NewTexture(im).(*ColorTexture)
	mipped := NewMipmappedTexture(im, MipTrilinear).(*ColorTexture)
	for i := 0; i < 1000; i++ {
		u := -2 + 4*float64(i)/999
		v := -2 + 4*float64((i*104729)%1000)/999
		if a, b := plain.Sample(u, v), mipped.Sample(u, v); a != b {
			t.Fatalf("Sample(%v, %v): plain %v != mipped %v", u, v, a, b)
		}
		if a, b := plain.BumpSample(u, v), mipped.BumpSample(u, v); a != b {
			t.Fatalf("BumpSample(%v, %v): plain %v != mipped %v", u, v, a, b)
		}
	}
}

func TestNonZeroBounds(t *testing.T) {
	im := image.NewRGBA(image.Rect(7, 3, 7+16, 3+8))
	for y := 3; y < 11; y++ {
		for x := 7; x < 23; x++ {
			im.SetRGBA(x, y, color.RGBA{uint8(x * 10), uint8(y * 20), 128, 255})
		}
	}
	tex := NewTexture(im).(*ColorTexture)
	if tex.Width != 16 || tex.Height != 8 {
		t.Fatalf("size = %dx%d, want 16x8", tex.Width, tex.Height)
	}
	want := NewColor(im.At(7, 3)).Pow(2.2)
	if got := tex.Data[0]; got != want {
		t.Fatalf("Data[0] = %v, want %v", got, want)
	}
	want = NewColor(im.At(22, 10)).Pow(2.2)
	if got := tex.Data[len(tex.Data)-1]; got != want {
		t.Fatalf("last texel = %v, want %v", got, want)
	}
}

func TestEmptyImage(t *testing.T) {
	for _, r := range []image.Rectangle{
		image.Rectangle{},
		image.Rect(5, 5, 5, 9),
		image.Rect(5, 5, 9, 5),
	} {
		tex := NewTexture(image.NewRGBA(r)).(*ColorTexture)
		c := tex.Sample(0.3, 0.7)
		if c != Black {
			t.Fatalf("empty image sample = %v, want black", c)
		}
		tex.BuildMipmaps(MipTrilinear)
		_ = tex.SampleFootprint(0.3, 0.7, 1000)
		_ = tex.NormalSample(0.3, 0.7)
		_ = tex.BumpSample(0.3, 0.7)
	}
}

func TestTinyTextures(t *testing.T) {
	for _, size := range [][2]int{{1, 1}, {1, 7}, {5, 1}, {2, 2}} {
		im := noiseImage(size[0], size[1], 3)
		tex := NewMipmappedTexture(im, MipTrilinear).(*ColorTexture)
		for i := 0; i < 100; i++ {
			u := -1 + 2*float64(i)/99
			v := float64(i) / 99
			if c := tex.Sample(u, v); !finiteColor(c) {
				t.Fatalf("size %v: Sample(%v, %v) = %v", size, u, v, c)
			}
			if c := tex.SampleFootprint(u, v, 64); !finiteColor(c) {
				t.Fatalf("size %v: SampleFootprint(%v, %v) = %v", size, u, v, c)
			}
			_ = tex.NormalSample(u, v)
			_ = tex.BumpSample(u, v)
			_ = tex.BumpSampleFootprint(u, v, 64)
		}
	}
	// A 1x1 texture returns its (linearized) color everywhere.
	im := image.NewRGBA(image.Rect(0, 0, 1, 1))
	im.SetRGBA(0, 0, color.RGBA{200, 100, 50, 255})
	tex := NewTexture(im).(*ColorTexture)
	want := NewColor(im.At(0, 0)).Pow(2.2)
	for _, uv := range [][2]float64{{0, 0}, {0.5, 0.5}, {0.999, 0.001}, {-3.7, 8.2}} {
		if got := tex.Sample(uv[0], uv[1]); !approxColor(got, want, 1e-9) {
			t.Fatalf("1x1 Sample(%v, %v) = %v, want %v", uv[0], uv[1], got, want)
		}
	}
}

func TestWrapAndNegativeUV(t *testing.T) {
	tex := NewTexture(noiseImage(32, 32, 4)).(*ColorTexture)
	for i := 0; i < 500; i++ {
		u := -4 + 8*float64(i)/499
		v := -4 + 8*float64((i*31)%500)/499
		// Integer UV shifts are not bit-exact in floating point; compare
		// with a tight epsilon instead.
		if a, b := tex.Sample(u, v), tex.Sample(u+3, v-2); !approxColor(a, b, 1e-9) {
			t.Fatalf("wrap mismatch: Sample(%v,%v)=%v != Sample(%v,%v)=%v", u, v, a, u+3, v-2, b)
		}
		if a, b := tex.Sample(u, v), tex.SampleFootprint(u+1, v+1, 0.5); !approxColor(a, b, 1e-9) {
			t.Fatalf("footprint<=1 mismatch: %v != %v", a, b)
		}
	}
}

func TestIntegerUVEdge(t *testing.T) {
	// Exact integer v wraps to 0 and the 1-v flip yields exactly 1, which
	// sent the original implementation out of bounds. The clamped corner
	// indices make this continuous with the v -> 1- limit.
	tex := NewMipmappedTexture(noiseImage(16, 16, 20), MipTrilinear).(*ColorTexture)
	for _, u := range []float64{0, 0.25, 0.5, 0.999} {
		for _, v := range []float64{-2, -1, 0, 1, 2, 100} {
			if c := tex.Sample(u, v); !finiteColor(c) {
				t.Fatalf("Sample(%v, %v) = %v", u, v, c)
			}
			if c := tex.SampleFootprint(u, v, 16); !finiteColor(c) {
				t.Fatalf("SampleFootprint(%v, %v) = %v", u, v, c)
			}
			// Integer v values all wrap to the same texel row, and the
			// v -> 1- limit matches the v -> 0- (negative) side.
			if got, want := tex.Sample(u, 0), tex.Sample(u, 1); got != want {
				t.Fatalf("u=%v: Sample v=0 %v != v=1 %v", u, got, want)
			}
			a := tex.Sample(u, -1e-9)
			b := tex.Sample(u, 1-1e-9)
			if !approxColor(a, b, 1e-6) {
				t.Fatalf("u=%v: v->0- %v not continuous with v->1- %v", u, a, b)
			}
		}
	}
}

func TestNonFiniteUV(t *testing.T) {
	tex := NewMipmappedTexture(noiseImage(16, 16, 5), MipTrilinear).(*ColorTexture)
	bad := []float64{math.NaN(), math.Inf(1), math.Inf(-1)}
	for _, u := range bad {
		for _, v := range bad {
			if c := tex.Sample(u, v); !finiteColor(c) {
				t.Fatalf("Sample(%v, %v) = %v", u, v, c)
			}
			if c := tex.SampleFootprint(u, v, 32); !finiteColor(c) {
				t.Fatalf("SampleFootprint(%v, %v) = %v", u, v, c)
			}
			if c := tex.SampleFootprint(0.5, 0.5, u); !finiteColor(c) {
				t.Fatalf("SampleFootprint footprint %v = %v", u, c)
			}
			_ = tex.NormalSample(u, v)
			_ = tex.BumpSample(u, v)
			_ = tex.BumpSampleFootprint(u, v, 32)
		}
	}
}

func TestMipChainDimensions(t *testing.T) {
	tex := NewMipmappedTexture(noiseImage(256, 256, 6), MipTrilinear).(*ColorTexture)
	if tex.MipLevels() != 8 {
		t.Fatalf("256x256 mip levels = %d, want 8", tex.MipLevels())
	}
	w, h, _ := tex.level(tex.MipLevels())
	if w != 1 || h != 1 {
		t.Fatalf("coarsest level = %dx%d, want 1x1", w, h)
	}

	// Non-power-of-two sizes halve with clamping down to 1x1.
	tex = NewMipmappedTexture(noiseImage(100, 37, 7), MipNearest).(*ColorTexture)
	want := [][2]int{{50, 18}, {25, 9}, {12, 4}, {6, 2}, {3, 1}, {1, 1}}
	if tex.MipLevels() != len(want) {
		t.Fatalf("100x37 mip levels = %d, want %d", tex.MipLevels(), len(want))
	}
	for i, wh := range want {
		w, h, _ := tex.level(i + 1)
		if w != wh[0] || h != wh[1] {
			t.Fatalf("level %d = %dx%d, want %dx%d", i+1, w, h, wh[0], wh[1])
		}
	}

	// 1x1 textures have no chain but remain sampleable.
	tex = NewMipmappedTexture(image.NewRGBA(image.Rect(0, 0, 1, 1)), MipTrilinear).(*ColorTexture)
	if tex.MipLevels() != 0 {
		t.Fatalf("1x1 mip levels = %d, want 0", tex.MipLevels())
	}
}

// TestMipEnergy verifies box-filtered mip levels preserve the mean (energy)
// of the base level for checkerboard, gradient and noise textures.
func TestMipEnergy(t *testing.T) {
	images := map[string]image.Image{
		"checker":  checkerImage(256, 1),
		"gradient": gradientImage(256, 64),
		"noise":    noiseImage(128, 128, 8),
	}
	for name, im := range images {
		tex := NewMipmappedTexture(im, MipTrilinear).(*ColorTexture)
		base := meanColor(tex.Data)
		for level := 1; level <= tex.MipLevels(); level++ {
			_, _, data := tex.level(level)
			if got := meanColor(data); !approxColor(got, base, 1e-9) {
				t.Fatalf("%s level %d mean = %v, want %v", name, level, got, base)
			}
		}
		// The 1x1 level equals the global average.
		_, _, data := tex.level(tex.MipLevels())
		if got := data[0]; !approxColor(got, base, 1e-9) {
			t.Fatalf("%s 1x1 level = %v, want %v", name, got, base)
		}
	}
	// A 1-pixel checkerboard averages to 50% gray in linear space.
	tex := NewMipmappedTexture(checkerImage(256, 1), MipTrilinear).(*ColorTexture)
	c := tex.SampleFootprint(0.3, 0.7, 1e9)
	if !approxColor(c, Color{0.5, 0.5, 0.5}, 1e-9) {
		t.Fatalf("checkerboard huge-footprint sample = %v, want 0.5 gray", c)
	}
}

// TestMipFlickerReduction compares point sampling against mip-filtered
// sampling under heavy minification: the filtered result must keep the mean
// (energy) while drastically reducing variance (flicker).
func TestMipFlickerReduction(t *testing.T) {
	for name, im := range map[string]image.Image{
		"checker": checkerImage(256, 1),
		"noise":   noiseImage(256, 256, 10),
	} {
		tex := NewMipmappedTexture(im, MipTrilinear).(*ColorTexture)
		const n = 4096
		varianceOf := func(sample func(u, v float64) Color) (float64, float64) {
			rnd := rand.New(rand.NewSource(11))
			var mean, m2 float64
			for i := 0; i < n; i++ {
				c := sample(rnd.Float64(), rnd.Float64())
				x := (c.R + c.G + c.B) / 3
				delta := x - mean
				mean += delta / float64(i+1)
				m2 += delta * (x - mean)
			}
			return mean, m2 / (n - 1)
		}
		baseMean, baseVar := varianceOf(tex.Sample)
		mipMean, mipVar := varianceOf(func(u, v float64) Color {
			return tex.SampleFootprint(u, v, 256)
		})
		if math.Abs(mipMean-baseMean) > 0.02 {
			t.Fatalf("%s: mip mean %v drifted from base mean %v", name, mipMean, baseMean)
		}
		if mipVar > baseVar*0.01 {
			t.Fatalf("%s: mip variance %v not << base variance %v", name, mipVar, baseVar)
		}
	}
}

// TestMipNearestVsTrilinear checks both level-selection strategies against
// hand-computed expectations.
func TestMipNearestVsTrilinear(t *testing.T) {
	im := noiseImage(8, 8, 12)
	nearest := NewMipmappedTexture(im, MipNearest).(*ColorTexture)
	tri := NewMipmappedTexture(im, MipTrilinear).(*ColorTexture)
	u, v := 0.31, 0.77
	wu, wv := wrapUV(u, v)

	// Nearest: lod 1.6 rounds to level 2.
	w, h, data := nearest.level(2)
	want := bilinear(w, h, data, wu, wv)
	if got := nearest.SampleLOD(u, v, 1.6); got != want {
		t.Fatalf("nearest lod 1.6 = %v, want level 2 sample %v", got, want)
	}
	// Trilinear: lod 1.5 blends levels 1 and 2 equally.
	w0, h0, d0 := tri.level(1)
	w1, h1, d1 := tri.level(2)
	c0 := bilinear(w0, h0, d0, wu, wv)
	c1 := bilinear(w1, h1, d1, wu, wv)
	want = c0.Mix(c1, 0.5)
	if got := tri.SampleLOD(u, v, 1.5); got != want {
		t.Fatalf("trilinear lod 1.5 = %v, want %v", got, want)
	}
	// lod beyond the chain clamps to the coarsest level.
	w, h, data = tri.level(tri.MipLevels())
	want = bilinear(w, h, data, wu, wv)
	if got := tri.SampleLOD(u, v, 1e9); got != want {
		t.Fatalf("clamped lod = %v, want coarsest %v", got, want)
	}
	// lod <= 0 is exactly the base-level Sample.
	if got := tri.SampleLOD(u, v, 0); got != tri.Sample(u, v) {
		t.Fatalf("lod 0 = %v, want Sample %v", got, tri.Sample(u, v))
	}
}

// TestBoundaryPixels samples along the wrap seams and verifies results stay
// finite, in gamut, and continuous across the boundary.
func TestBoundaryPixels(t *testing.T) {
	tex := NewMipmappedTexture(checkerImage(64, 4), MipTrilinear).(*ColorTexture)
	for _, footprint := range []float64{0, 1, 8, 64, 1e6} {
		for i := 0; i <= 256; i++ {
			u := float64(i) / 256
			for _, v := range []float64{0, 1e-9, 0.5, 1 - 1e-9} {
				c := tex.SampleFootprint(u, v, footprint)
				if !finiteColor(c) {
					t.Fatalf("footprint %v: SampleFootprint(%v, %v) = %v", footprint, u, v, c)
				}
				if c.R < -1e-9 || c.R > 1+1e-9 {
					t.Fatalf("footprint %v: out-of-gamut %v at (%v, %v)", footprint, c, u, v)
				}
			}
		}
		// Wrap continuity across the u and v seams.
		for _, v := range []float64{0.13, 0.87} {
			a := tex.SampleFootprint(-1e-9, v, footprint)
			b := tex.SampleFootprint(1-1e-9, v, footprint)
			if !approxColor(a, b, 1e-6) {
				t.Fatalf("footprint %v: u seam %v != %v", footprint, a, b)
			}
		}
		for _, u := range []float64{0.13, 0.87} {
			a := tex.SampleFootprint(u, -1e-9, footprint)
			b := tex.SampleFootprint(u, 1-1e-9, footprint)
			if !approxColor(a, b, 1e-6) {
				t.Fatalf("footprint %v: v seam %v != %v", footprint, a, b)
			}
		}
	}
}

func TestMemoryCaps(t *testing.T) {
	oldTex, oldMip := MaxTextureTexels, MaxMipTexels
	defer func() { MaxTextureTexels, MaxMipTexels = oldTex, oldMip }()

	// Oversized base images are downsampled until they fit.
	MaxTextureTexels = 100
	im := image.NewRGBA(image.Rect(0, 0, 64, 64))
	for i := range im.Pix {
		im.Pix[i] = 200
	}
	tex := NewTexture(im).(*ColorTexture)
	if tex.Width != 8 || tex.Height != 8 {
		t.Fatalf("capped texture = %dx%d, want 8x8", tex.Width, tex.Height)
	}
	if c := tex.Sample(0.3, 0.6); !approxColor(c, Color{200.0 / 255, 200.0 / 255, 200.0 / 255}.Pow(2.2), 1e-9) {
		t.Fatalf("capped texture color = %v", c)
	}

	// The mip chain truncates at the texel budget and sampling still works.
	MaxTextureTexels = 1 << 28
	MaxMipTexels = 64*64 + 32*32
	tex = NewMipmappedTexture(noiseImage(64, 64, 13), MipTrilinear).(*ColorTexture)
	if tex.MipLevels() != 1 {
		t.Fatalf("truncated chain levels = %d, want 1", tex.MipLevels())
	}
	if c := tex.SampleFootprint(0.4, 0.4, 1e9); !finiteColor(c) {
		t.Fatalf("truncated chain sample = %v", c)
	}
}

func TestInputImageUnmodified(t *testing.T) {
	im := noiseImage(64, 64, 14)
	before := make([]byte, len(im.Pix))
	copy(before, im.Pix)
	tex := NewMipmappedTexture(im, MipTrilinear)
	tex.Pow(2)
	tex.MulScalar(0.5)
	_ = tex.Sample(0.5, 0.5)
	for i := range im.Pix {
		if im.Pix[i] != before[i] {
			t.Fatalf("input image modified at byte %d", i)
		}
	}
}

func TestPowMulScalarAffectMipChain(t *testing.T) {
	a := NewMipmappedTexture(noiseImage(16, 16, 15), MipTrilinear).(*ColorTexture)
	b := NewMipmappedTexture(noiseImage(16, 16, 15), MipTrilinear).(*ColorTexture)
	b.Pow(2).MulScalar(0.5)
	for level := 0; level <= a.MipLevels(); level++ {
		_, _, da := a.level(level)
		_, _, db := b.level(level)
		for i := range da {
			want := da[i].Pow(2).MulScalar(0.5)
			if db[i] != want {
				t.Fatalf("level %d texel %d = %v, want %v", level, i, db[i], want)
			}
		}
	}
}

// TestMaterialAtFootprint verifies the material-level entry points share the
// mip strategy and that footprint 0 reproduces MaterialAt exactly.
func TestMaterialAtFootprint(t *testing.T) {
	tex := NewMipmappedTexture(checkerImage(64, 1), MipTrilinear)
	material := DiffuseMaterial(White)
	material.Texture = tex
	material.GlossTexture = tex
	shape := NewSphere(Vector{}, 1, material)
	point := Vector{0, 0, 1}
	m0 := MaterialAt(shape, point)
	m1 := MaterialAtFootprint(shape, point, 0)
	if m0.Color != m1.Color || m0.Gloss != m1.Gloss {
		t.Fatalf("footprint 0 mismatch: %v vs %v", m0, m1)
	}
	m2 := MaterialAtFootprint(shape, point, 1e9)
	if !approxColor(m2.Color, Color{0.5, 0.5, 0.5}, 1e-9) {
		t.Fatalf("minified material color = %v, want 0.5 gray", m2.Color)
	}
	if math.Abs(m2.Gloss-0.5) > 1e-9 {
		t.Fatalf("minified gloss = %v, want 0.5", m2.Gloss)
	}
}

func BenchmarkSampleBilinear(b *testing.B) {
	tex := NewTexture(noiseImage(512, 512, 16)).(*ColorTexture)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tex.Sample(float64(i%997)/997, float64(i%991)/991)
	}
}

func BenchmarkSampleMipNearest(b *testing.B) {
	tex := NewMipmappedTexture(noiseImage(512, 512, 17), MipNearest).(*ColorTexture)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tex.SampleFootprint(float64(i%997)/997, float64(i%991)/991, 32)
	}
}

func BenchmarkSampleMipTrilinear(b *testing.B) {
	tex := NewMipmappedTexture(noiseImage(512, 512, 18), MipTrilinear).(*ColorTexture)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tex.SampleFootprint(float64(i%997)/997, float64(i%991)/991, 32)
	}
}

func BenchmarkBuildMipmaps(b *testing.B) {
	tex := NewTexture(noiseImage(512, 512, 19)).(*ColorTexture)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tex.BuildMipmaps(MipTrilinear)
	}
}
