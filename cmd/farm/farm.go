// Command farm renders one image across independent machines (or processes)
// without changing the result. It has three subcommands:
//
//	farm plan   -print the shard list for a render
//	farm render -shard N -out shard-N.png   render one shard
//	farm merge  -out final.png shard-*.png  merge rendered shards
//
// Every shard uses the same deterministic Sobol configuration, keyed by the
// seed and the absolute pixel coordinate, so the merged image is identical to
// rendering the whole frame on one machine. Run farm with a fixed scene name
// on each worker; the scene lives in farm_scene.go so the geometry and camera
// are compiled identically everywhere.
package main

import (
	"flag"
	"fmt"
	"image"
	"log"
	"os"

	"github.com/fogleman/pt/pt"
)

func main() {
	log.SetFlags(0)
	if len(os.Args) < 2 {
		usage()
	}
	switch os.Args[1] {
	case "plan":
		cmdPlan(os.Args[2:])
	case "render":
		cmdRender(os.Args[2:])
	case "merge":
		cmdMerge(os.Args[2:])
	default:
		usage()
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: farm plan|render|merge [flags]")
	fmt.Fprintln(os.Stderr, "  farm plan   -scene NAME -w W -h H -shards N")
	fmt.Fprintln(os.Stderr, "  farm render -scene NAME -shard N -out PATH [-w W -h H -shards N -seed S -spp N -iteration I]")
	fmt.Fprintln(os.Stderr, "  farm merge  -out PATH shard1.png shard2.png ...")
	os.Exit(2)
}

func cmdPlan(args []string) {
	fs := flag.NewFlagSet("plan", flag.ExitOnError)
	sceneName := fs.String("scene", "cornell", "scene name")
	w := fs.Int("w", 160, "image width")
	h := fs.Int("h", 120, "image height")
	shards := fs.Int("shards", 4, "number of shards")
	wide := fs.Bool("wide", false, "use a 2D shard grid instead of horizontal bands")
	fs.Parse(args)
	if *sceneName == "cornell" {
		// Keep the Cornell example's real resolution default sane.
		if *w == 160 && *h == 120 {
			*w, *h = 512, 512
		}
	}
	var rects []pt.ShardMeta
	if *wide {
		rects = pt.ShardRectsWide(*w, *h, *shards)
	} else {
		rects = pt.ShardRects(*w, *h, *shards)
	}
	if len(rects) == 0 {
		log.Fatalf("no shards for %dx%d", *w, *h)
	}
	rects[0].ImageW, rects[0].ImageH = *w, *h
	for i, r := range rects {
		fmt.Printf("%d %d %d %d %d\n", i, r.MinX, r.MinY, r.MaxX, r.MaxY)
	}
}

func cmdRender(args []string) {
	fs := flag.NewFlagSet("render", flag.ExitOnError)
	sceneName := fs.String("scene", "cornell", "scene name")
	shard := fs.Int("shard", 0, "shard index from farm plan")
	w := fs.Int("w", 160, "image width")
	h := fs.Int("h", 120, "image height")
	shards := fs.Int("shards", 4, "total shards")
	wide := fs.Bool("wide", false, "use a 2D shard grid")
	seed := fs.Uint64("seed", 1, "Sobol seed")
	spp := fs.Int("spp", -1, "samples per pixel (negative = random subsampling count)")
	iteration := fs.Int("iteration", 0, "iteration index for seed separation")
	out := fs.String("out", "shard.png", "output shard path (a .json sidecar is written too)")
	quiet := fs.Bool("quiet", true, "suppress render banner")
	fs.Parse(args)
	if *sceneName == "cornell" {
		if *w == 160 && *h == 120 {
			*w, *h = 512, 512
		}
	}
	scene, camera, sampler, err := buildScene(*sceneName, *spp)
	if err != nil {
		log.Fatalf("scene %q: %v", *sceneName, err)
	}
	var rects []pt.ShardMeta
	if *wide {
		rects = pt.ShardRectsWide(*w, *h, *shards)
	} else {
		rects = pt.ShardRects(*w, *h, *shards)
	}
	if *shard < 0 || *shard >= len(rects) {
		log.Fatalf("shard %d out of range [0,%d)", *shard, len(rects))
	}
	m := rects[*shard]
	m.Mode = pt.SampleModeSobol.String()
	m.Seed = *seed
	m.Iteration = *iteration
	m.SamplesPerPixel = *spp
	m.ImageW, m.ImageH = *w, *h

	region := image.Rect(m.MinX, m.MinY, m.MaxX, m.MaxY)
	cfg := pt.RenderConfig{
		Mode:    pt.SampleModeSobol,
		Seed:    *seed,
		Region:  &region,
		Quiet:   *quiet,
		Workers: 0,
	}
	buf := pt.RenderToBuffer(scene, camera, sampler, *w, *h, *spp, cfg)
	if err := pt.SaveShardPNG(*out, buf.Image(pt.ColorChannel), m); err != nil {
		log.Fatalf("save shard: %v", err)
	}
	fmt.Printf("wrote %s (region %d,%d - %d,%d)\n", *out, m.MinX, m.MinY, m.MaxX, m.MaxY)
}

func cmdMerge(args []string) {
	fs := flag.NewFlagSet("merge", flag.ExitOnError)
	out := fs.String("out", "merged.png", "output image path")
	fs.Parse(args)
	paths := fs.Args()
	if len(paths) == 0 {
		log.Fatal("merge: no shard paths given")
	}
	var mes []pt.ShardMeta
	var ims []image.Image
	for _, path := range paths {
		im, err := pt.LoadImage(path)
		if err != nil {
			log.Fatalf("load %s: %v", path, err)
		}
		meta, err := pt.ReadShardMeta(path + ".json")
		if err != nil {
			log.Fatalf("load metadata %s.json: %v", path, err)
		}
		ims = append(ims, im)
		mes = append(mes, meta)
	}
	merged, err := pt.MergeShards(mes, ims)
	if err != nil {
		log.Fatalf("merge: %v", err)
	}
	if err := pt.SavePNG(*out, merged); err != nil {
		log.Fatalf("save %s: %v", *out, err)
	}
	fmt.Printf("wrote %s from %d shards\n", *out, len(paths))
}
