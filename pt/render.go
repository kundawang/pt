package pt

import (
	"context"
	"fmt"
	"image"
	"runtime"
	"strings"
	"sync"
	"time"
)

func showProgress(start time.Time, rays uint64, i, h int) {
	pct := int(100 * float64(i) / float64(h))
	elapsed := time.Since(start)
	rps := float64(rays) / elapsed.Seconds()
	fmt.Printf("\r%4d / %d (%3d%%) [", i, h, pct)
	for p := 0; p < 100; p += 3 {
		if pct > p {
			fmt.Print("=")
		} else {
			fmt.Print(" ")
		}
	}
	fmt.Printf("] %s %s ", DurationString(elapsed), NumberString(rps))
}

// render renders one pass over buf using the tiled scheduler. It preserves
// the historical console output and GOMAXPROCS behavior.
func render(scene *Scene, camera *Camera, sampler Sampler, samplesPerPixel int, buf *Buffer) {
	_ = renderContext(context.Background(), scene, camera, sampler, samplesPerPixel, buf, RenderConfig{})
}

func renderContext(ctx context.Context, scene *Scene, camera *Camera, sampler Sampler, samplesPerPixel int, buf *Buffer, config RenderConfig) error {
	w, h := buf.W, buf.H
	ncpu := runtime.NumCPU()
	if ncpu < 1 {
		ncpu = 1
	}
	runtime.GOMAXPROCS(ncpu)
	scene.Compile()
	absSamples := samplesPerPixel
	if absSamples < 0 {
		absSamples = -absSamples
	}
	fmt.Printf("%d x %d pixels, %d spp, %d cores\n", w, h, absSamples, ncpu)
	start := time.Now()
	scene.rays = 0
	if config.Workers == 0 {
		config.Workers = ncpu
	}
	progress := config.Progress
	config.Progress = func(done, total int) {
		showProgress(start, scene.RayCount(), done, total)
		if progress != nil {
			progress(done, total)
		}
	}
	r := NewRenderer(scene, camera, sampler, samplesPerPixel, config)
	total := len(r.Tiles(w, h))
	if total > 0 {
		showProgress(start, scene.RayCount(), 0, total)
	}
	err := r.Render(ctx, buf)
	fmt.Println()
	return err
}

func writeImage(ctx context.Context, path string, buf *Buffer, channel Channel, wg *sync.WaitGroup) {
	defer wg.Done()
	if ctx.Err() != nil {
		return
	}
	im := buf.Image(channel)
	if ctx.Err() != nil {
		return
	}
	if err := SavePNG(path, im); err != nil {
		panic(err)
	}
}

func Render(scene *Scene, camera *Camera, sampler Sampler, w, h, samplesPerPixel int) image.Image {
	buf := NewBuffer(w, h)
	render(scene, camera, sampler, samplesPerPixel, buf)
	return buf.Image(ColorChannel)
}

// RenderContext renders a single pass into a new buffer using the tiled
// scheduler, honoring ctx cancellation. The image returned alongside a
// non-nil error contains whatever samples completed before cancellation.
func RenderContext(ctx context.Context, scene *Scene, camera *Camera, sampler Sampler, w, h, samplesPerPixel int, config RenderConfig) (image.Image, error) {
	buf := NewBuffer(w, h)
	err := renderContext(ctx, scene, camera, sampler, samplesPerPixel, buf, config)
	return buf.Image(ColorChannel), err
}

func IterativeRender(pathTemplate string, iterations int, scene *Scene, camera *Camera, sampler Sampler, w, h, samplesPerPixel int) image.Image {
	im, _ := IterativeRenderContext(context.Background(), pathTemplate, iterations, scene, camera, sampler, w, h, samplesPerPixel, RenderConfig{})
	return im
}

// IterativeRenderContext behaves like IterativeRender but stops scheduling
// new iterations — and skips pending image writes — when ctx is cancelled.
// It always waits for in-flight renders and writes to finish before
// returning, so no goroutines are leaked.
func IterativeRenderContext(ctx context.Context, pathTemplate string, iterations int, scene *Scene, camera *Camera, sampler Sampler, w, h, samplesPerPixel int, config RenderConfig) (image.Image, error) {
	var wg sync.WaitGroup
	scene.Compile()
	buf := NewBuffer(w, h)
	var err error
	for i := 1; i <= iterations; i++ {
		if ctx.Err() != nil {
			err = ctx.Err()
			break
		}
		fmt.Printf("\n[Iteration %d of %d]\n", i, iterations)
		if e := renderContext(ctx, scene, camera, sampler, samplesPerPixel, buf, config); e != nil {
			err = e
			break
		}
		path := pathTemplate
		if strings.Contains(path, "%") {
			path = fmt.Sprintf(pathTemplate, i)
		}
		bufCopy := buf.Copy()
		wg.Add(1)
		go writeImage(ctx, path, bufCopy, ColorChannel, &wg)
		wg.Add(1)
		go writeImage(ctx, "deviation.png", bufCopy, StandardDeviationChannel, &wg)
	}
	wg.Wait()
	return buf.Image(ColorChannel), err
}

func ChannelRender(scene *Scene, camera *Camera, sampler Sampler, w, h, samplesPerPixel int) <-chan image.Image {
	ch := make(chan image.Image)
	go func() {
		scene.Compile()
		buf := NewBuffer(w, h)
		for i := 1; ; i++ {
			render(scene, camera, sampler, samplesPerPixel, buf)
			ch <- buf.Image(ColorChannel)
		}
	}()
	return ch
}
