#!/usr/bin/env python
"""
Copy right belongs to Zhuang Lab, Harvard University.
Classes that handles reading STORM movie files. Currently this
is limited to the dax, fits, spe and tif formats.

Hazen 06/13
"""

import hashlib
import numpy
import os
import re
import tifffile


def inferReader(filename, verbose = False):
    """
    Given a file name this will try to return the appropriate
    reader based on the file extension.
    """
    ext = os.path.splitext(filename)[1]
    if (ext == ".dax"):
        return DaxReader(filename, verbose = verbose)
    elif (ext == ".fits"):
        return FITSReader(filename, verbose = verbose)
    elif (ext == ".spe"):
        return SpeReader(filename, verbose = verbose)
    elif (ext == ".tif") or (ext == ".tiff"):
        return TifReader(filename, verbose = verbose)
    else:
        print(ext, "is not a recognized file type")
        raise IOError("only .dax, .spe and .tif are supported (case sensitive..)")


class Reader(object):
    """
    The superclass containing those functions that
    are common to reading a STORM movie file.

    Subclasses should implement:
     1. __init__(self, filename, verbose = False)
        This function should open the file and extract the
        various key bits of meta-data such as the size in XY
        and the length of the movie.

     2. loadAFrame(self, frame_number)
        Load the requested frame and return it as numpy array.
    """
    def __init__(self, filename, verbose = False):
        super(Reader, self).__init__()
        self.filename = filename
        self.fileptr = None
        self.verbose = verbose

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, value, traceback):
        self.close()

    def averageFrames(self, start = False, end = False):
        """
        Average multiple frames in a movie.
        """
        if (not start):
            start = 0
        if (not end):
            end = self.number_frames 

        length = end - start
        average = numpy.zeros((self.image_height, self.image_width), numpy.float)
        for i in range(length):
            if self.verbose and ((i%10)==0):
                print(" processing frame:", i, " of", self.number_frames)
            average += self.loadAFrame(i + start)
            
        average = average/float(length)
        return average

    def close(self):
        if self.fileptr is not None:
            self.fileptr.close()
            self.fileptr = None
        
    def filmFilename(self):
        """
        Returns the film name.
        """
        return self.filename

    def filmSize(self):
        """
        Returns the film size.
        """
        return [self.image_width, self.image_height, self.number_frames]

    def filmLocation(self):
        """
        Returns the picture x,y location, if available.
        """
        if hasattr(self, "stage_x"):
            return [self.stage_x, self.stage_y]
        else:
            return [0.0, 0.0]

    def filmScale(self):
        """
        Returns the scale used to display the film when
        the picture was taken.
        """
        if hasattr(self, "scalemin") and hasattr(self, "scalemax"):
            return [self.scalemin, self.scalemax]
        else:
            return [100, 2000]
    def hashID(self):
        """
        A (hopefully) unique string that identifies this movie.
        """
        return hashlib.md5(self.loadAFrame(0).tostring()).hexdigest()

    def loadAFrame(self, frame_number):
        assert frame_number >= 0, "Frame_number must be greater than or equal to 0, it is " + str(frame_number)
        assert frame_number < self.number_frames, "Frame number must be less than " + str(self.number_frames)

    def lockTarget(self):
        """
        Returns the film focus lock target.
        """
        if hasattr(self, "lock_target"):
            return self.lock_target
        else:
            return 0.0


class DaxReader(Reader):
    """
    Dax reader class. This is a Zhuang lab custom format.
    """
    def __init__(self, filename, verbose = False):
        super(DaxReader, self).__init__(filename, verbose = verbose)
        # save the filenames
        dirname = os.path.dirname(filename)
        if (len(dirname) > 0):
            dirname = dirname + "/"
        self.inf_filename = dirname + os.path.splitext(os.path.basename(filename))[0] + ".inf"

        # defaults
        self.image_height = None
        self.image_width = None

        # extract the movie information from the associated inf file
        size_re = re.compile(r'frame dimensions = ([\d]+) x ([\d]+)')
        length_re = re.compile(r'number of frames = ([\d]+)')
        endian_re = re.compile(r' (big|little) endian')
        stagex_re = re.compile(r'Stage X = ([\d\.\-]+)')
        stagey_re = re.compile(r'Stage Y = ([\d\.\-]+)')
        lock_target_re = re.compile(r'Lock Target = ([\d\.\-]+)')
        scalemax_re = re.compile(r'scalemax = ([\d\.\-]+)')
        scalemin_re = re.compile(r'scalemin = ([\d\.\-]+)')

        inf_file = open(self.inf_filename, "r")
        while 1:
            line = inf_file.readline()
            if not line: break
            m = size_re.match(line)
            if m:
                self.image_height = int(m.group(2))
                self.image_width = int(m.group(1))
            m = length_re.match(line)
            if m:
                self.number_frames = int(m.group(1))
            m = endian_re.search(line)
            if m:
                if m.group(1) == "big":
                    self.bigendian = 1
                else:
                    self.bigendian = 0
            m = stagex_re.match(line)
            if m:
                self.stage_x = float(m.group(1))
            m = stagey_re.match(line)
            if m:
                self.stage_y = float(m.group(1))
            m = lock_target_re.match(line)
            if m:
                self.lock_target = float(m.group(1))
            m = scalemax_re.match(line)
            if m:
                self.scalemax = int(m.group(1))
            m = scalemin_re.match(line)
            if m:
                self.scalemin = int(m.group(1))

        inf_file.close()

        # set defaults, probably correct, but warn the user 
        # that they couldn't be determined from the inf file.
        if not self.image_height:
            print("Could not determine image size, assuming 256x256.")
            self.image_height = 256
            self.image_width = 256

        # open the dax file
        if os.path.exists(filename):
            self.fileptr = open(filename, "rb")
        else:
            if self.verbose:
                print("dax data not found", filename)

    def loadAFrame(self, frame_number):
        """
        Load a frame & return it as a numpy array.
        """
        super(DaxReader, self).loadAFrame(frame_number)

        self.fileptr.seek(frame_number * self.image_height * self.image_width * 2)
        image_data = numpy.fromfile(self.fileptr, dtype='uint16', count = self.image_height * self.image_width)
        image_data = numpy.reshape(image_data, [self.image_height, self.image_width])
        if self.bigendian:
            image_data.byteswap(True)
        return image_data


class FITSReader(Reader):
    """
    FITS file reader class.

    FIXME: This depends on internals of astropy.io.fits that I'm sure
           we are not supposed to be messing with. The problem is that
           astropy.io.fits does not support memmap'd images when the
           image is scaled (which is pretty much always the case?). To
           get around this we set _ImageBaseHDU._do_not_scale_image_data
           to True, then do the image scaling ourselves.

           We want memmap = True as generally it won't make sense to
           load the entire movie into memory.

           Another consequence of this is that we only support
           'pseudo unsigned' 16 bit FITS format files.
    """
    def __init__(self, filename, verbose = False):
        super(FITSReader, self).__init__(filename, verbose = verbose)

        # Import here to avoid making astropy mandatory for everybody.
        from astropy.io import fits

        self.hdul = fits.open(filename, memmap = True)

        hdr = self.hdul[0].header
        # We only handle 16 bit FITS files.
        assert ((hdr['BITPIX'] == 16) and (hdr['bscale'] == 1) and (hdr['bzero'] == 32768)),\
            "Only 16 bit pseudo-unsigned FITS format is currently supported!"

        # Get image size. We're assuming that the film is a data cube in
        # the first / primary HDU.
        #
        self.image_height = hdr['naxis2']
        self.image_width = hdr['naxis1']
        if (hdr['naxis'] == 3):
            self.number_frames = hdr['naxis3']
        else:
            self.number_frames = 1

        self.hdu0 = self.hdul[0]
        # Hack, override astropy.io.fits internal so that we can load
        # data with memmap = True.
        #
        self.hdu0._do_not_scale_image_data = True

    def close(self):
        pass

    def loadAFrame(self, frame_number):
        super(FITSReader, self).loadAFrame(frame_number)

        frame = self.hdu0.data[frame_number,:,:].astype(numpy.uint16)
        frame -= 32768
        return frame


class SpeReader(Reader):
    """
    SPE (Roper Scientific) reader class.
    """
    def __init__(self, filename, verbose = False):
        super(SpeReader, self).__init__(filename, verbose = verbose)

        # open the file & read the header
        self.header_size = 4100
        self.fileptr = open(filename, "rb")

        self.fileptr.seek(42)
        self.image_width = int(numpy.fromfile(self.fileptr, numpy.uint16, 1)[0])
        self.fileptr.seek(656)
        self.image_height = int(numpy.fromfile(self.fileptr, numpy.uint16, 1)[0])
        self.fileptr.seek(1446)
        self.number_frames = int(numpy.fromfile(self.fileptr, numpy.uint32, 1)[0])

        self.fileptr.seek(108)
        image_mode = int(numpy.fromfile(self.fileptr, numpy.uint16, 1)[0])
        if (image_mode == 0):
            self.image_size = 4 * self.image_width * self.image_height
            self.image_mode = numpy.float32
        elif (image_mode == 1):
            self.image_size = 4 * self.image_width * self.image_height
            self.image_mode = numpy.uint32
        elif (image_mode == 2):
            self.image_size = 2 * self.image_width * self.image_height
            self.image_mode = numpy.int16
        elif (image_mode == 3):
            self.image_size = 2 * self.image_width * self.image_height
            self.image_mode = numpy.uint16
        else:
            print("unrecognized spe image format: ", image_mode)

    def loadAFrame(self, frame_number, cast_to_int16 = True):
        """
        Load a frame & return it as a numpy array.
        """
        super(SpeReader, self).loadAFrame(frame_number)
        self.fileptr.seek(self.header_size + frame_number * self.image_size)
        image_data = numpy.fromfile(self.fileptr, dtype=self.image_mode, count = self.image_height * self.image_width)
        if cast_to_int16:
            image_data = image_data.astype(numpy.uint16)
        image_data = numpy.reshape(image_data, [self.image_height, self.image_width])
        return image_data


class TifReader(Reader):
    """
    TIF reader class.
    When given tiff files with multiple pages and multiple frames per
    page this is just going to read the file as if it was one long movie.
    """
    def __init__(self, filename, verbose = False):
        super(TifReader, self).__init__(filename, verbose)

        # Save the filename
        self.fileptr = tifffile.TiffFile(filename)
        number_pages = len(self.fileptr.pages)

        # Get shape by loading first frame
        self.isize = self.fileptr.asarray(key=0).shape

        # Check if each page has multiple frames.
        if (len(self.isize) == 3):
            self.frames_per_page = self.isize[0]
            self.image_height = self.isize[1]
            self.image_width = self.isize[2]
        else:
            self.frames_per_page = 1
            self.image_height = self.isize[0]
            self.image_width = self.isize[1]

        if self.verbose:
            print("{0:0d} frames per page, {1:0d} pages".format(self.frames_per_page, number_pages))
        self.number_frames = self.frames_per_page * number_pages
        self.page_number = -1
        self.page_data = None

    def loadAFrame(self, frame_number, cast_to_int16 = True):
        super(TifReader, self).loadAFrame(frame_number)

        # Load the right frame from the right page.
        if (self.frames_per_page > 1):
            page = int(frame_number/self.frames_per_page)
            frame = frame_number % self.frames_per_page

            # This is an optimization for files with a large number of frames
            # per page. In this case tifffile will keep loading the entire
            # page over and over again, which really slows everything down.
            # Ideally tifffile would let us specify which frame on the page
            # we wanted.
            #
            # Since it was going to load the whole thing anyway we'll have
            # memory overflow either way, so not much we can do about that
            # except hope for small file sizes.
            #
            if (page != self.page_number):
                self.page_data = self.fileptr.asarray(key = page)
                self.page_number = page
            image_data = self.page_data[frame,:,:]
        else:
            image_data = self.fileptr.asarray(key = frame_number)
            assert (len(image_data.shape) == 2), "not a monochrome tif image."
        if cast_to_int16:
            image_data = image_data.astype(numpy.uint16)
        return image_data


if (__name__ == "__main__"):

    import sys

    if (len(sys.argv) != 2):
        print("usage: <movie>")
        exit()

    movie = inferReader(sys.argv[1], verbose = True)
    print("Movie size is", movie.filmSize())

    frame = movie.loadAFrame(0)
    print(frame.shape, type(frame), frame.dtype)

#
# The MIT License
#
# Copyright (c) 2013 Zhuang Lab, Harvard University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
