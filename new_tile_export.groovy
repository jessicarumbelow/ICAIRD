import qupath.lib.images.servers.LabeledImageServer

print "Getting current image data..."
def imageData = getCurrentImageData()

// Define output path (relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName()).split('_')[0]

def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'new_exported_tiles',name)
mkdirs(pathOutput)

double avg_pixel_size = imageData.getServer().getPixelCalibration().getAveragedPixelSize()
print avg_pixel_size

double downsample = 1
print downsample

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .useDetections()
    .useCells()
    .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
    .addLabel('CD8', 1)      // Choose output labels (the order matters!)
    .addLabel('CD3', 2)
    .addLabel('CD20', 3)
    .addLabel('CD8: CD3', 4)
    .addLabel('CD20: CD3', 5)
    .addLabel('CD20: CD8', 6)
    .addLabel('CD20: CD8: CD3', 7)
    .multichannelOutput(false)  // If true, each label is a different channel (required for multiclass probability)
    .build()

// Create an exporter that requests corresponding tiles from the original & labeled image servers
new TileExporter(imageData)
    .downsample(downsample)     // Define export resolution
    .imageExtension('.tif')     // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    .tileSize(256)              // Define size of each tile, in pixels
    .labeledServer(labelServer) // Define the labeled image server to use (i.e. the one we just built)
    .annotatedTilesOnly(true)  // If true, only export tiles if there is a (labeled) annotation present
    .overlap(0)                // Define overlap, in pixel units at the export resolution
    .writeTiles(pathOutput)     // Write tiles to the specified directory

print pathOutput
