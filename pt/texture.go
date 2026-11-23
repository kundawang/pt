package pt

import (
	"fmt"
	"image"
	"math"
)

type Texture interface {
	Sample(u, v float64) Color
	NormalSample(u, v float64) Vector
	BumpSample(u, v float64) Vector
	Pow(a float64) Texture
	MulScalar(a float64) Texture
}

// FootprintTexture is an optional interface implemented by textures that can
// pre-filter themselves for a footprint measured in base-level texels. It is
// used to fight aliasing (flicker) when a texture is minified.
type FootprintTexture interface {
	Texture
	SampleFootprint(u, v, footprint float64) Color
	NormalSampleFootprint(u, v, footprint float64) Vector
	BumpSampleFootprint(u, v, footprint float64) Vector
}

// SampleWithFootprint samples t at (u, v). If t implements FootprintTexture
// the sample is filtered for the given footprint; otherwise it falls back to
// t.Sample. A footprint of one texel or less always reproduces t.Sample.
func SampleWithFootprint(t Texture, u, v, footprint float64) Color {
	if ft, ok := t.(FootprintTexture); ok {
		return ft.SampleFootprint(u, v, footprint)
	}
	return t.Sample(u, v)
}

// NormalWithFootprint is SampleWithFootprint for normal maps.
func NormalWithFootprint(t Texture, u, v, footprint float64) Vector {
	if ft, ok := t.(FootprintTexture); ok {
		return ft.NormalSampleFootprint(u, v, footprint)
	}
	return t.NormalSample(u, v)
}

// BumpWithFootprint is SampleWithFootprint for bump maps.
func BumpWithFootprint(t Texture, u, v, footprint float64) Vector {
	if ft, ok := t.(FootprintTexture); ok {
		return ft.BumpSampleFootprint(u, v, footprint)
	}
	return t.BumpSample(u, v)
}

// MipFilter selects how a mip level is derived from a texture footprint.
type MipFilter int

const (
	// MipNearest samples the single mip level nearest to the ideal level.
	MipNearest MipFilter = iota
	// MipTrilinear blends the two mip levels bracketing the ideal level.
	MipTrilinear
)

// Memory safety caps. They are variables so tests and embedders can tune
// them; both are generous by default and only guard against pathological
// inputs.
var (
	// MaxTextureTexels caps the number of texels in the base level. Images
	// with more texels are box-downsampled (in linear space) until they fit.
	MaxTextureTexels = 1 << 28
	// MaxMipTexels caps the total number of texels stored in a mip chain,
	// including the base level. The chain is truncated once the cap is
	// reached; sampling then clamps to the coarsest available level.
	MaxMipTexels = 1 << 28
)

var textures map[string]Texture

func init() {
	textures = make(map[string]Texture)
}

func GetTexture(path string) Texture {
	if texture, ok := textures[path]; ok {
		return texture
	}
	if texture, err := LoadTexture(path); err == nil {
		textures[path] = texture
		return texture
	}
	return nil
}

func LoadTexture(path string) (Texture, error) {
	fmt.Printf("Loading IMG: %s\n", path)
	im, err := LoadImage(path)
	if err != nil {
		return nil, err
	}
	return NewTexture(im), nil
}

type ColorTexture struct {
	Width  int
	Height int
	Data   []Color

	// Filter selects the mip filtering mode used by SampleLOD and the
	// footprint-based samplers. It has no effect unless a mip chain has
	// been built with BuildMipmaps.
	Filter MipFilter

	levels []mipLevel // mip levels below the base level, nil when disabled
}

type mipLevel struct {
	width  int
	height int
	data   []Color
}

// NewTexture creates a ColorTexture from an image. Pixel data is copied and
// gamma-decoded to linear space (Pow 2.2); the source image is never
// modified. Empty images fall back to a 1x1 black texture and oversized
// images are downsampled to respect MaxTextureTexels, so the result is
// always safe to sample.
func NewTexture(im image.Image) Texture {
	bounds := im.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w < 1 || h < 1 {
		return &ColorTexture{1, 1, []Color{Black}, MipTrilinear, nil}
	}
	data := make([]Color, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			index := y*w + x
			data[index] = NewColor(im.At(bounds.Min.X+x, bounds.Min.Y+y)).Pow(2.2)
		}
	}
	w, h, data = fitTextureMemory(w, h, data)
	return &ColorTexture{w, h, data, MipTrilinear, nil}
}

// NewMipmappedTexture creates a ColorTexture with a mip chain using the
// given filter. Without the chain the texture behaves exactly like one from
// NewTexture.
func NewMipmappedTexture(im image.Image, filter MipFilter) Texture {
	t := NewTexture(im).(*ColorTexture)
	t.BuildMipmaps(filter)
	return t
}

// fitTextureMemory downsamples a base level until it fits MaxTextureTexels.
func fitTextureMemory(w, h int, data []Color) (int, int, []Color) {
	for int64(w)*int64(h) > int64(MaxTextureTexels) && (w > 1 || h > 1) {
		w, h, data = downsampleLevel(w, h, data)
	}
	return w, h, data
}

// downsampleLevel halves a level with a 2x2 box filter, clamping at odd
// edges. Averaging is done in linear space so energy is preserved.
func downsampleLevel(w, h int, data []Color) (int, int, []Color) {
	nw := w / 2
	if nw < 1 {
		nw = 1
	}
	nh := h / 2
	if nh < 1 {
		nh = 1
	}
	out := make([]Color, nw*nh)
	for y := 0; y < nh; y++ {
		for x := 0; x < nw; x++ {
			x0 := 2 * x
			y0 := 2 * y
			x1 := x0 + 1
			if x1 >= w {
				x1 = w - 1
			}
			y1 := y0 + 1
			if y1 >= h {
				y1 = h - 1
			}
			c := data[y0*w+x0].
				Add(data[y0*w+x1]).
				Add(data[y1*w+x0]).
				Add(data[y1*w+x1])
			out[y*nw+x] = c.MulScalar(0.25)
		}
	}
	return nw, nh, out
}

// BuildMipmaps builds the optional mip chain below the base level and sets
// the filter used by the footprint-based samplers. Levels are 2x2
// box-filtered in linear space down to 1x1; the chain is truncated if it
// would exceed MaxMipTexels, in which case sampling clamps to the coarsest
// available level. Rebuilding replaces any existing chain.
func (t *ColorTexture) BuildMipmaps(filter MipFilter) {
	t.Filter = filter
	t.levels = nil
	total := int64(t.Width) * int64(t.Height)
	w, h, data := t.Width, t.Height, t.Data
	for w > 1 || h > 1 {
		nw, nh, next := downsampleLevel(w, h, data)
		total += int64(nw) * int64(nh)
		if total > int64(MaxMipTexels) {
			break
		}
		t.levels = append(t.levels, mipLevel{nw, nh, next})
		w, h, data = nw, nh, next
	}
}

// MipLevels returns the number of mip levels below the base level.
func (t *ColorTexture) MipLevels() int {
	return len(t.levels)
}

func (t *ColorTexture) Pow(a float64) Texture {
	for i := range t.Data {
		t.Data[i] = t.Data[i].Pow(a)
	}
	for _, l := range t.levels {
		for i := range l.data {
			l.data[i] = l.data[i].Pow(a)
		}
	}
	return t
}

func (t *ColorTexture) MulScalar(a float64) Texture {
	for i := range t.Data {
		t.Data[i] = t.Data[i].MulScalar(a)
	}
	for _, l := range t.levels {
		for i := range l.data {
			l.data[i] = l.data[i].MulScalar(a)
		}
	}
	return t
}

// wrapUV applies the texture's repeat wrapping and v-flip. Non-finite
// coordinates map to 0 so that indexing always stays in bounds.
func wrapUV(u, v float64) (float64, float64) {
	if math.IsNaN(u) || math.IsInf(u, 0) {
		u = 0
	}
	if math.IsNaN(v) || math.IsInf(v, 0) {
		v = 0
	}
	u = Fract(Fract(u) + 1)
	v = Fract(Fract(v) + 1)
	return u, 1 - v
}

// bilinear samples a level with bilinear interpolation. Corner indices are
// clamped so degenerate (1-pixel-wide or 1-pixel-tall) levels stay in
// bounds; for levels at least 2 texels in both dimensions this is exactly
// the original sampling math.
func bilinear(w, h int, data []Color, u, v float64) Color {
	fx := float64(w) - 1
	fy := float64(h) - 1
	X, x := math.Modf(u * fx)
	Y, y := math.Modf(v * fy)
	x0 := int(X)
	y0 := int(Y)
	x1 := x0 + 1
	if x1 >= w {
		x1 = w - 1
	}
	y1 := y0 + 1
	if y1 >= h {
		y1 = h - 1
	}
	c00 := data[y0*w+x0]
	c01 := data[y1*w+x0]
	c10 := data[y0*w+x1]
	c11 := data[y1*w+x1]
	c := Black
	c = c.Add(c00.MulScalar((1 - x) * (1 - y)))
	c = c.Add(c10.MulScalar(x * (1 - y)))
	c = c.Add(c01.MulScalar((1 - x) * y))
	c = c.Add(c11.MulScalar(x * y))
	return c
}

func (t *ColorTexture) bilinearSample(u, v float64) Color {
	return bilinear(t.Width, t.Height, t.Data, u, v)
}

// level returns the dimensions and data of a mip level; level 0 is the base
// level and out-of-range levels clamp to the base or coarsest level.
func (t *ColorTexture) level(level int) (int, int, []Color) {
	if level <= 0 || level > len(t.levels) {
		return t.Width, t.Height, t.Data
	}
	l := t.levels[level-1]
	return l.width, l.height, l.data
}

// selectLevel maps an LOD to the nearest available mip level. It is shared
// by all footprint-aware samplers so color, gloss, normal and bump textures
// pick levels consistently.
func (t *ColorTexture) selectLevel(lod float64) int {
	if len(t.levels) == 0 || lod <= 0 || math.IsNaN(lod) {
		return 0
	}
	level := int(lod + 0.5)
	if level > len(t.levels) {
		level = len(t.levels)
	}
	return level
}

// lodForFootprint converts a footprint, measured in base-level texels, to a
// level of detail. Footprints of one texel or less (magnification) and
// non-finite footprints map to the base level.
func lodForFootprint(footprint float64) float64 {
	if footprint <= 1 || math.IsNaN(footprint) {
		return 0
	}
	return math.Log2(footprint)
}

func (t *ColorTexture) Sample(u, v float64) Color {
	u, v = wrapUV(u, v)
	return t.bilinearSample(u, v)
}

// SampleLOD samples the texture at an explicit level of detail using the
// texture's MipFilter. Without a mip chain, or at lod <= 0, it is identical
// to Sample. The lod is clamped to the coarsest available level.
func (t *ColorTexture) SampleLOD(u, v, lod float64) Color {
	u, v = wrapUV(u, v)
	if len(t.levels) == 0 || lod <= 0 || math.IsNaN(lod) {
		return t.bilinearSample(u, v)
	}
	if t.Filter == MipNearest {
		w, h, data := t.level(t.selectLevel(lod))
		return bilinear(w, h, data, u, v)
	}
	maxLOD := float64(len(t.levels))
	if lod > maxLOD {
		lod = maxLOD
	}
	lo := int(math.Floor(lod))
	hi := lo + 1
	if hi > len(t.levels) {
		hi = len(t.levels)
	}
	f := lod - float64(lo)
	w0, h0, d0 := t.level(lo)
	w1, h1, d1 := t.level(hi)
	c0 := bilinear(w0, h0, d0, u, v)
	c1 := bilinear(w1, h1, d1, u, v)
	return c0.Mix(c1, f)
}

// SampleFootprint samples the texture pre-filtered for a footprint measured
// in base-level texels, e.g. the number of texels covered by one pixel.
func (t *ColorTexture) SampleFootprint(u, v, footprint float64) Color {
	return t.SampleLOD(u, v, lodForFootprint(footprint))
}

func (t *ColorTexture) NormalSample(u, v float64) Vector {
	c := t.Sample(u, v)
	return Vector{c.R*2 - 1, c.G*2 - 1, c.B*2 - 1}.Normalize()
}

// NormalSampleFootprint is NormalSample pre-filtered for a footprint.
func (t *ColorTexture) NormalSampleFootprint(u, v, footprint float64) Vector {
	c := t.SampleFootprint(u, v, footprint)
	return Vector{c.R*2 - 1, c.G*2 - 1, c.B*2 - 1}.Normalize()
}

// bumpSample computes a finite-difference bump vector on one mip level.
func bumpSample(w, h int, data []Color, u, v float64) Vector {
	x := ClampInt(int(u*float64(w)), 0, w-1)
	y := ClampInt(int(v*float64(h)), 0, h-1)
	x1, x2 := ClampInt(x-1, 0, w-1), ClampInt(x+1, 0, w-1)
	y1, y2 := ClampInt(y-1, 0, h-1), ClampInt(y+1, 0, h-1)
	cx := data[y*w+x1].Sub(data[y*w+x2])
	cy := data[y1*w+x].Sub(data[y2*w+x])
	return Vector{cx.R, cy.R, 0}
}

func (t *ColorTexture) BumpSample(u, v float64) Vector {
	u, v = wrapUV(u, v)
	return bumpSample(t.Width, t.Height, t.Data, u, v)
}

// BumpSampleFootprint is BumpSample computed on the mip level selected for
// the given footprint, using the same level-selection strategy as the color
// samplers.
func (t *ColorTexture) BumpSampleFootprint(u, v, footprint float64) Vector {
	u, v = wrapUV(u, v)
	level := t.selectLevel(lodForFootprint(footprint))
	w, h, data := t.level(level)
	return bumpSample(w, h, data, u, v)
}
