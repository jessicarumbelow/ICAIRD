/**
 * Script to export binary masks corresponding to all annotations of an image,
 * optionally along with extracted image regions.
 *
 * Note: Pay attention to the 'downsample' value to control the export resolution!
 *
 * @author Pete Bankhead
 */

import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject

import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage

// Get the main QuPath data structures
def imageData = getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()

def cd8_cd3lo = getCellObjects().findAll{it.getPathClass() == getPathClass("CD8")}
def cd8_cd3hi = getCellObjects().findAll{it.getPathClass() == getPathClass("CD8: CD3")}
def cd3 = getCellObjects().findAll{it.getPathClass() == getPathClass("CD3")}
def cd20 = getCellObjects().findAll{it.getPathClass() == getPathClass("CD20")}

def len = cd8_cd3lo.size() + cd8_cd3hi.size() + cd3.size() + cd20.size()

def other = getCellObjects().findAll{!(it.getPathClass() in [getPathClass("CD8"),getPathClass("CD8: CD3"),getPathClass
("CD3"),getPathClass("C20")])}.shuffled()[0..len]

cls = [other, cd8_cd3lo, cd8_cd3hi, cd3, cd20]
clsNames = ['OTHER', 'CD8_CD3LO', 'CD8_CD3HI', 'CD3', 'CD20']

// Define downsample value for export resolution & output directory, creating directory if necessary
def downsample = 1.0
def pathOutput = buildFilePath(QPEx.PROJECT_BASE_DIR,'exported_cells',GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName()).split('_')[0])
mkdirs(pathOutput)

// Define image export type; valid values are JPG, PNG or null (if no image region should be exported with the mask)
// Note: masks will always be exported as PNG
def imageExportType = 'PNG'

// Export each annotation

for (c in 0..5) {
    clsName = clsNames[c]
    cls[c].each {
        saveImageAndMask(pathOutput, server, it, downsample, imageExportType, clsName)
        }
    }
    print 'Done!'

/**
 * Save extracted image region & mask corresponding to an object ROI.
 *
 * @param pathOutput Directory in which to store the output
 * @param server ImageServer for the relevant image
 * @param pathObject The object to export
 * @param downsample Downsample value for the export of both image region & mask
 * @param imageExportType Type of image (original pixels, not mask!) to export ('JPG', 'PNG' or null)
 * @return
 */
def saveImageAndMask(String pathOutput, ImageServer server, PathObject pathObject, double downsample, String
imageExportType, String className) {
    // Extract ROI & classification name
    def roi = pathObject.getROI()
    def pathClass = pathObject.getPathClass()
    if (roi == null) {
        print 'Warning! No ROI for object ' + pathObject + ' - cannot export corresponding region & mask'
        return
    }

    // Create a region from the ROI
    def region = RegionRequest.createInstance(server.getPath(), downsample, roi)

    // Create a name
    String name = String.format('%s-%s-(%.2f,%d,%d,%d,%d)',
            GeneralTools.getNameWithoutExtension(server.getMetadata().getName()),
            className,
            region.getDownsample(),
            region.getX(),
            region.getY(),
            region.getWidth(),
            region.getHeight()
    )

    // Request the BufferedImage
    def img = server.readBufferedImage(region)

    // Create a mask using Java2D functionality
    // (This involves applying a transform to a graphics object, so that none needs to be applied to the ROI coordinates)
    def shape = RoiTools.getShape(roi)
    def imgMask = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_BYTE_GRAY)
    def g2d = imgMask.createGraphics()
    g2d.setColor(Color.WHITE)
    g2d.scale(1.0/downsample, 1.0/downsample)
    g2d.translate(-region.getX(), -region.getY())
    g2d.fill(shape)
    g2d.dispose()
    // Create filename & export
    if (imageExportType != null) {
        print name
        def ImagePath = buildFilePath(pathOutput, name + '.tif')

        writeImageRegion(server, region, ImagePath)
    }
    // Export the mask
    def fileMask = new File(pathOutput, name + '-mask.png')
    ImageIO.write(imgMask, 'PNG', fileMask)

}

