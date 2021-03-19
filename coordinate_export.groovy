import qupath.lib.images.servers.LabeledImageServer

print "Getting current image data..."
def imageData = getCurrentImageData()

// Define output path (relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName()).split('_')[0]
print(name)

def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'new_exported_coords')
mkdirs(pathOutput)

saveDetectionMeasurements(pathOutput, "Class", "Centroid X µm", "Centroid Y µm")