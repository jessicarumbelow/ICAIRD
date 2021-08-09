

import qupath.lib.common.ColorTools
import qupath.lib.objects.classes.PathClass
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.RoiTools
import qupath.lib.gui.scripting.QPEx

import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.awt.image.IndexColorModel
import qupath.lib.gui.scripting.QPEx


double requestedPixelSizeMicrons = 4

int maxTileSize = 256

def pathOutput = QPEx.buildFilePath(QPEx.PROJECT_BASE_DIR, 'exported_tiles')
QPEx.mkdirs(pathOutput)

def imageData = QPEx.getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()

def detections = hierarchy.getFlattenedObjectList(null).findAll {
    it.isDetection() && it.hasROI() && (it.getPathClass() != null) }

def pathClasses = detections.collect({it.getPathClass()}) as Set

def labelKey = new StringBuilder()
def pathClassColors = [:]
int n = pathClasses.size() + 1
def r = ([(byte)0] * n) as byte[]
def g = r.clone()
def b = r.clone()
def a = r.clone()
pathClasses.eachWithIndex{ PathClass entry, int i ->
    int label = i+1
    String name = entry == null ? 'None' : entry.toString()
    labelKey << name << '\t' << label << System.lineSeparator()
    pathClassColors.put(entry, new Color(label, label, label))
    int rgb = entry == null ? ColorTools.makeRGB(127, 127, 127) : entry.getColor()
    r[label] = ColorTools.red(rgb)
    g[label] = ColorTools.green(rgb)
    b[label] = ColorTools.blue(rgb)
    a[label] = 255
}

double downsample = 1
if (requestedPixelSizeMicrons > 0)
    downsample = requestedPixelSizeMicrons / server.getPixelCalibration().getAveragedPixelSizeMicrons()

int spacing = (int)(maxTileSize * downsample)

def requests = new ArrayList<RegionRequest>()
for (int y = 0; y < server.getHeight(); y += spacing) {
    int h = spacing
    if (y + h > server.getHeight())
        h = server.getHeight() - y
    for (int x = 0; x < server.getWidth(); x += spacing) {
        int w = spacing
        if (x + w > server.getWidth())
            w = server.getWidth() - x
        requests << RegionRequest.createInstance(server.getPath(), downsample, x, y, w, h)
    }
}


def keyName = String.format('%s_(downsample=%.3f,tiles=%d)-key.txt', server.getPath().split('/')[-1].split('.czi')[0], downsample, maxTileSize)
def fileLabels = new File(pathOutput, keyName)
fileLabels.text = labelKey.toString()

requests.parallelStream().forEach { request ->
    String name = String.format('%s_(%.2f,%d,%d,%d,%d)',
            server.getPath().split('/')[-1].split('.czi')[0],
            request.getDownsample(),
            request.getX(),
            request.getY(),
            request.getWidth(),
            request.getHeight()
    )

    def img = server.readBufferedImage(request)
    width = img.getWidth()
    height = img.getHeight()

    def imgMask = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
    def g2d = imgMask.createGraphics()
    g2d.setClip(0, 0, width, height)
    g2d.scale(1.0/downsample, 1.0/downsample)
    g2d.translate(-request.getX(), -request.getY())
    int count = 0
    for (detection in detections) {
        def roi = detection.getROI()
        if (!request.intersects(roi.getBoundsX(), roi.getBoundsY(), roi.getBoundsWidth(), roi.getBoundsHeight()))
            continue
        def shape = RoiTools.getShape(roi)
        def color = pathClassColors.get(detection.getPathClass())
        g2d.setColor(color)
        g2d.fill(shape)
        count++
    }
    g2d.dispose()
    if (count > 0) {
        def buf = imgMask.getRaster().getDataBuffer() as DataBufferByte
        def bytes = buf.getData()
        if (!bytes.any { it != (byte)0 })
            return
        def colorModel = new IndexColorModel(8, n, r, g, b, a)
        def imgMaskColor = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_INDEXED, colorModel)
        System.arraycopy(bytes, 0, imgMaskColor.getRaster().getDataBuffer().getData(), 0, width*height)
        imgMask = imgMaskColor

        def fileOutput = new File(pathOutput, name + '_mask.png')
        ImageIO.write(imgMask, 'PNG', fileOutput)

        def imgOutput = new File(pathOutput, name + '.png')
        ImageIO.write(img, 'PNG', imgOutput)
    }
}
print 'Done!'