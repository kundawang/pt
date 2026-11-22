package pt

import (
	"image"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestRegionRenderMatchesWholeRender is the render-farm guarantee: splitting
// one deterministic render into per-machine regions and merging must be
// bit-identical to rendering it whole.
func TestRegionRenderMatchesWholeRender(t *testing.T) {
	const w, h = 20, 14
	base := deterministicConfig(123)
	whole := renderTestBuffer(t, w, h, base)

	shards := ShardRects(w, h, 4)
	if len(shards) < 2 {
		t.Fatalf("expected multiple shards, got %d", len(shards))
	}
	var ims []image.Image
	for _, m := range shards {
		m.Mode = base.Mode.String()
		m.Seed = base.Seed
		m.SamplesPerPixel = 4
		region := image.Rect(m.MinX, m.MinY, m.MaxX, m.MaxY)
		cfg := base
		cfg.Region = &region
		buf := renderTestBuffer(t, w, h, cfg)
		ims = append(ims, buf.Image(ColorChannel))
	}
	merged, err := MergeShards(shards, ims)
	if err != nil {
		t.Fatalf("MergeShards: %v", err)
	}
	wholeIm := whole.Image(ColorChannel)
	if diff, err := ImageMeanAbsDiff(merged, wholeIm); err != nil {
		t.Fatal(err)
	} else if diff != 0 {
		t.Fatalf("merged region render differs from whole render by %v", diff)
	}
	// Byte-level equality of the canvas is the strongest statement.
	rb, wb := merged.Bounds(), wholeIm.Bounds()
	if rb != wb {
		t.Fatalf("bounds %v != %v", rb, wb)
	}
	for y := wb.Min.Y; y < wb.Max.Y; y++ {
		for x := wb.Min.X; x < wb.Max.X; x++ {
			r1, g1, b1, a1 := merged.At(x, y).RGBA()
			r2, g2, b2, a2 := wholeIm.At(x, y).RGBA()
			if r1 != r2 || g1 != g2 || b1 != b2 || a1 != a2 {
				t.Fatalf("pixel (%d,%d) differs after region merge", x, y)
			}
		}
	}
}

func TestRegionClipsToBounds(t *testing.T) {
	const w, h = 10, 8
	cfg := deterministicConfig(1)
	// Region partially outside and fully outside must not panic.
	for _, r := range []image.Rectangle{
		image.Rect(-5, -5, 4, 3),
		image.Rect(w+1, 0, w+10, h),
		image.Rect(5, 5, 2, 2), // inverted -> empty
	} {
		region := r
		cfg.Region = &region
		_ = renderTestBuffer(t, w, h, cfg)
	}
}

func TestShardRectsCoverage(t *testing.T) {
	for _, tc := range []struct{ w, h, n int }{
		{20, 14, 1}, {20, 14, 4}, {20, 14, 7}, {3, 3, 100}, {1, 1, 1},
	} {
		mes := ShardRects(tc.w, tc.h, tc.n)
		cover := make(map[image.Point]int)
		for _, m := range mes {
			if m.MinX < 0 || m.MinY < 0 || m.MaxX > tc.w || m.MaxY > tc.h {
				t.Fatalf("%v: shard %v out of bounds", tc, m)
			}
			for y := m.MinY; y < m.MaxY; y++ {
				for x := m.MinX; x < m.MaxX; x++ {
					cover[image.Pt(x, y)]++
				}
			}
		}
		if len(cover) != tc.w*tc.h {
			t.Fatalf("%v: covered %d/%d pixels", tc, len(cover), tc.w*tc.h)
		}
		for p, n := range cover {
			if n != 1 {
				t.Fatalf("%v: pixel %v covered %d times", tc, p, n)
			}
		}
	}
}

func TestMergeShardsRejectsMismatch(t *testing.T) {
	const w, h = 8, 4
	mes := ShardRects(w, h, 2)
	mes[0].Seed, mes[1].Seed = 1, 2
	ims := []image.Image{image.NewRGBA(image.Rect(0, 0, w, h)), image.NewRGBA(image.Rect(0, 0, w, h))}
	if _, err := MergeShards(mes, ims); err == nil {
		t.Fatal("expected seed mismatch error")
	}

	// Overlapping shards are rejected.
	overlap := []ShardMeta{
		{ImageW: w, ImageH: h, MinX: 0, MinY: 0, MaxX: w, MaxY: 2},
		{ImageW: w, ImageH: h, MinX: 0, MinY: 1, MaxX: w, MaxY: h},
	}
	if _, err := MergeShards(overlap, ims); err == nil {
		t.Fatal("expected overlap error")
	}

	// Uncovered pixels are rejected.
	gap := []ShardMeta{{ImageW: w, ImageH: h, MinX: 0, MinY: 0, MaxX: w, MaxY: 1}}
	if _, err := MergeShards(gap, ims[:1]); err == nil {
		t.Fatal("expected uncovered pixel error")
	}

	// Empty input is rejected.
	if _, err := MergeShards(nil, nil); err == nil {
		t.Fatal("expected empty merge error")
	}
}

func TestShardSaveLoadRoundTrip(t *testing.T) {
	const w, h = 12, 9
	shards := ShardRects(w, h, 3)
	for i := range shards {
		shards[i].Mode = "sobol"
		shards[i].Seed = 55
		shards[i].SamplesPerPixel = 3
	}
	base := deterministicConfig(55)
	dir := t.TempDir()
	paths := make([]string, len(shards))
	for i, m := range shards {
		region := image.Rect(m.MinX, m.MinY, m.MaxX, m.MaxY)
		cfg := base
		cfg.Region = &region
		im := renderTestBuffer(t, w, h, cfg).Image(ColorChannel)
		path := filepath.Join(dir, "shard"+string(rune('0'+i))+".png")
		if err := SaveShardPNG(path, im, m); err != nil {
			t.Fatalf("SaveShardPNG: %v", err)
		}
		paths[i] = path
	}
	var ims []image.Image
	var mes []ShardMeta
	for _, p := range paths {
		im, err := LoadImage(p)
		if err != nil {
			t.Fatal(err)
		}
		meta, err := ReadShardMeta(p + ".json")
		if err != nil {
			t.Fatal(err)
		}
		ims = append(ims, im)
		mes = append(mes, meta)
	}
	merged, err := MergeShards(mes, ims)
	if err != nil {
		t.Fatalf("MergeShards: %v", err)
	}
	// The round-tripped PNG stores 16-bit RGBA; compare against the same
	// whole-image render to confirm the region split survives serialization.
	whole := renderTestBuffer(t, w, h, base).Image(ColorChannel)
	if diff, err := ImageMeanAbsDiff(merged, whole); err != nil {
		t.Fatal(err)
	} else if diff > 1e-3 {
		t.Fatalf("serialized shard merge differs from whole by %v", diff)
	}
	if _, err := os.Stat(paths[0] + ".json"); err != nil {
		t.Fatalf("metadata not written: %v", err)
	}
}

func TestImageMeanAbsDiff(t *testing.T) {
	a := image.NewRGBA(image.Rect(0, 0, 2, 2))
	b := image.NewRGBA(image.Rect(0, 0, 2, 2))
	if d, err := ImageMeanAbsDiff(a, b); err != nil || d != 0 {
		t.Fatalf("identical images diff = %v, %v", d, err)
	}
	b.Set(0, 0, White.RGBA())
	d, err := ImageMeanAbsDiff(a, b)
	if err != nil {
		t.Fatal(err)
	}
	want := 1.0 / 4.0 // one of four pixels, all channels differing by 1
	if math.Abs(d-want) > 1e-9 {
		t.Fatalf("diff = %v, want %v", d, want)
	}
	c := image.NewRGBA(image.Rect(0, 0, 3, 2))
	if _, err := ImageMeanAbsDiff(a, c); err == nil {
		t.Fatal("expected size mismatch error")
	}
}
