package pt

import (
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
)

// ShardMeta describes one region of a render that was produced independently,
// possibly on another machine. It is written alongside each shard so the
// coordinator can verify that all shards agree on the render parameters
// before merging them.
type ShardMeta struct {
	// ImageW and ImageH are the full image dimensions every shard shares.
	ImageW, ImageH int
	// MinX, MinY, MaxX, MaxY describe the shard's region in half-open pixel
	// coordinates.
	MinX, MinY, MaxX, MaxY int
	// Mode, Seed, Iteration and SamplesPerPixel identify the sampling
	// configuration. In Sobol mode a merge is only bit-reproducible if every
	// shard agrees on all four.
	Mode            string `json:"mode"`
	Seed            uint64 `json:"seed"`
	Iteration       int    `json:"iteration"`
	SamplesPerPixel int    `json:"samples_per_pixel"`
}

// SaveShardPNG writes a shard's region to path. The image must be the full
// image size; only the metadata region is cropped out, so any shard can be
// merged back onto the full canvas. Shards are encoded at 16 bits per channel
// so serializing them does not lose precision relative to the in-memory
// buffer. An empty or out-of-bounds region is an error rather than a silently
// empty file.
func SaveShardPNG(path string, im image.Image, meta ShardMeta) error {
	if meta.ImageW != im.Bounds().Dx() || meta.ImageH != im.Bounds().Dy() {
		return fmt.Errorf("pt: shard image is %dx%d but metadata says %dx%d",
			im.Bounds().Dx(), im.Bounds().Dy(), meta.ImageW, meta.ImageH)
	}
	r := image.Rect(meta.MinX, meta.MinY, meta.MaxX, meta.MaxY)
	if !r.In(im.Bounds()) {
		return fmt.Errorf("pt: shard region %v outside image bounds %v", r, im.Bounds())
	}
	if r.Empty() {
		return errors.New("pt: shard region is empty")
	}
	cropped := image.NewRGBA64(image.Rect(0, 0, r.Dx(), r.Dy()))
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			cropped.Set(x-r.Min.X, y-r.Min.Y, im.At(x, y))
		}
	}
	if err := SavePNG(path, cropped); err != nil {
		return err
	}
	return WriteShardMeta(path+".json", meta)
}

// WriteShardMeta serializes meta as indented JSON.
func WriteShardMeta(path string, meta ShardMeta) error {
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0644)
}

// ReadShardMeta loads a shard metadata file.
func ReadShardMeta(path string) (ShardMeta, error) {
	var meta ShardMeta
	data, err := os.ReadFile(path)
	if err != nil {
		return meta, err
	}
	err = json.Unmarshal(data, &meta)
	return meta, err
}

// MergeShards combines independently rendered shard images onto one canvas.
// The images may be either full-size canvases with only the shard region
// filled, or the cropped images written by SaveShardPNG; the metadata region
// says where each belongs. Every pixel in the target must be written exactly
// once, otherwise an error reports the uncovered or doubly covered pixels
// instead of silently merging a corrupt image. Merge output is bit-identical
// to a single-machine render when the shards used the same parameters.
func MergeShards(mes []ShardMeta, ims []image.Image) (image.Image, error) {
	if len(mes) != len(ims) {
		return nil, fmt.Errorf("pt: %d shard metadata entries for %d images", len(mes), len(ims))
	}
	if len(mes) == 0 {
		return nil, errors.New("pt: no shards to merge")
	}
	w, h := mes[0].ImageW, mes[0].ImageH
	if w <= 0 || h <= 0 {
		return nil, fmt.Errorf("pt: invalid image size %dx%d", w, h)
	}
	for i, m := range mes {
		if m.ImageW != w || m.ImageH != h {
			return nil, fmt.Errorf("pt: shard %d size %dx%d != %dx%d", i, m.ImageW, m.ImageH, w, h)
		}
		if m.Mode != mes[0].Mode || m.Seed != mes[0].Seed || m.Iteration != mes[0].Iteration ||
			m.SamplesPerPixel != mes[0].SamplesPerPixel {
			return nil, fmt.Errorf("pt: shard %d params %s/seed=%d/iter=%d/spp=%d do not match shard 0",
				i, m.Mode, m.Seed, m.Iteration, m.SamplesPerPixel)
		}
	}
	coverage := make([]uint8, w*h)
	result := image.NewRGBA64(image.Rect(0, 0, w, h))
	for i, m := range mes {
		r := image.Rect(m.MinX, m.MinY, m.MaxX, m.MaxY)
		if !r.In(result.Bounds()) {
			return nil, fmt.Errorf("pt: shard %d region %v outside %dx%d image", i, r, w, h)
		}
		if r.Empty() {
			return nil, fmt.Errorf("pt: shard %d region is empty", i)
		}
		b := ims[i].Bounds()
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				idx := y*w + x
				coverage[idx]++
				if coverage[idx] > 1 {
					return nil, fmt.Errorf("pt: pixel (%d,%d) covered by more than one shard", x, y)
				}
				// Full-size canvases are indexed absolutely; cropped shards
				// are indexed from their own origin.
				sx, sy := x, y
				if !image.Pt(x, y).In(b) {
					sx = b.Min.X + (x - r.Min.X)
					sy = b.Min.Y + (y - r.Min.Y)
				}
				result.Set(x, y, ims[i].At(sx, sy))
			}
		}
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if coverage[y*w+x] == 0 {
				return nil, fmt.Errorf("pt: pixel (%d,%d) not covered by any shard", x, y)
			}
		}
	}
	return result, nil
}

// ShardRects partitions a w x h image into n non-overlapping shards, row-major.
// It is the canonical farm split; clients render one rectangle each and the
// coordinator merges them with MergeShards. Because Sobol sample values key on
// absolute pixel coordinates, this split does not affect the merged image.
func ShardRects(w, h, n int) []ShardMeta {
	if n < 1 {
		n = 1
	}
	if n > w*h {
		n = w * h
	}
	// Chunk rows so every shard is a contiguous band.
	rowsPer := (h + n - 1) / n
	if rowsPer < 1 {
		rowsPer = 1
	}
	var result []ShardMeta
	for y := 0; y < h; y += rowsPer {
		y1 := y + rowsPer
		if y1 > h {
			y1 = h
		}
		result = append(result, ShardMeta{ImageW: w, ImageH: h, MinX: 0, MinY: y, MaxX: w, MaxY: y1})
	}
	return result
}

// ShardRectsWide partitions a w x h image into n non-overlapping shards using
// a two-dimensional grid rather than horizontal bands. Wide images or many
// machines get better cache and network balance. Because Sobol samples key on
// absolute pixel coordinates, the grid shape cannot change the merged image.
func ShardRectsWide(w, h, n int) []ShardMeta {
	if n < 1 {
		n = 1
	}
	if n > w*h {
		n = w * h
	}
	targetX := int(math.Round(math.Sqrt(float64(n) * float64(w) / float64(h))))
	if targetX < 1 {
		targetX = 1
	}
	if targetX > n {
		targetX = n
	}
	targetY := (n + targetX - 1) / targetX
	if targetX > w {
		targetX = w
	}
	if targetY > h {
		targetY = h
	}
	var result []ShardMeta
	for ty := 0; ty < targetY; ty++ {
		y0 := ty * h / targetY
		y1 := (ty + 1) * h / targetY
		for tx := 0; tx < targetX; tx++ {
			x0 := tx * w / targetX
			x1 := (tx + 1) * w / targetX
			if x1 > x0 && y1 > y0 {
				result = append(result, ShardMeta{ImageW: w, ImageH: h, MinX: x0, MinY: y0, MaxX: x1, MaxY: y1})
			}
		}
	}
	return result
}

// ImageMeanAbsDiff returns the mean absolute per-channel difference in
// [0,1] between two images whose bounds must match. It is the error metric
// used to compare reproducible renders at slightly different configurations.
func ImageMeanAbsDiff(a, b image.Image) (float64, error) {
	if a.Bounds().Size() != b.Bounds().Size() {
		return 0, fmt.Errorf("pt: image sizes differ: %v vs %v", a.Bounds().Size(), b.Bounds().Size())
	}
	bounds := a.Bounds()
	var sum float64
	n := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			ca := color.NRGBAModel.Convert(a.At(x, y)).(color.NRGBA)
			cb := color.NRGBAModel.Convert(b.At(x, y)).(color.NRGBA)
			sum += float64(absDiff(ca.R, cb.R)+absDiff(ca.G, cb.G)+absDiff(ca.B, cb.B)) / 255
			n += 3
		}
	}
	if n == 0 {
		return 0, nil
	}
	return sum / float64(n), nil
}

func absDiff(a, b uint8) int {
	if a > b {
		return int(a - b)
	}
	return int(b - a)
}
