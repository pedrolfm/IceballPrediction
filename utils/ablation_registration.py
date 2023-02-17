#! /usr/bin/python

import argparse, sys, shutil, os, logging
import SimpleITK as sitk
import json
#import sitkUtils


def command_iteration(method):
    #print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f} : {method.GetOptimizerPosition()}")
    pass


def resampleImage(srcImage, refImage, transform, interp='linear'):
    
    dimension = srcImage.GetDimension()
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(refImage)
    if interp == 'nearest':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    resampler.SetOutputSpacing(refImage.GetSpacing())
    resampler.SetSize(refImage.GetSize())
    resampler.SetOutputDirection(srcImage.GetDirection())
    resampler.SetOutputOrigin(srcImage.GetOrigin())
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetDefaultPixelValue(0)
    
    dstImage = resampler.Execute(srcImage)
    
    return dstImage


def mask(image, anatomLabel, dilation=0):
    
    relabelMap =  { i : 0 for i in range(2, 256) }
    maskLabel = sitk.ChangeLabel(anatomLabel, changeMap=relabelMap)

    if dilation > 0:
        filter = sitk.BinaryDilateImageFilter()
        filter.SetKernelRadius ( dilation )
        filter.SetForegroundValue ( 1 )
        maskLabel = filter.Execute ( maskLabel )
        
    #sitk.WriteImage(maskLabel, 'masklabel.nrrd')
    
    # Make sure that the mask and the moving image occupy the same physical space.
    transform = sitk.AffineTransform(3)
    maskLabel = resampleImage(maskLabel, image, transform, 'nearest')
    
    maskedImage = sitk.Mask(image, maskLabel, maskingValue=0, outsideValue=1)

    return maskedImage


def createMaskFromAnatomLabel(anatomLabel, image, dilation=0):
    
    relabelMap =  { i : 0 for i in range(2, 256) }
    maskLabel = sitk.ChangeLabel(anatomLabel, changeMap=relabelMap)

    if dilation > 0:
        filter = sitk.BinaryDilateImageFilter()
        filter.SetKernelRadius ( dilation )
        filter.SetForegroundValue ( 1 )
        maskLabel = filter.Execute ( maskLabel )
        
    #sitk.WriteImage(maskLabel, 'masklabel.nrrd')
    
    # Make sure that the mask and the moving image occupy the same physical space.
    transform = sitk.AffineTransform(3)
    maskLabel = resampleImage(maskLabel, image, transform, 'nearest')
    
    return maskLabel


def registerImages(fixedImage, movingImage, param, mask=None, maskType='moving'):
  # maskType specifies which image will be masked. It should be either 'moving' or 'fixed'.

    if not 'numberOfBins' in param:
        param['numberOfBins'] = 50
    if not 'samplingPercentage' in param:
        param['samplingPercentage'] = 0.02

    offset = None
    if 'initialOffset' in param:
        offset = param['initialOffset']
            
    R = sitk.ImageRegistrationMethod()

    # set up registration method

    rigid_versor_trans = sitk.VersorRigid3DTransform()
    if offset:
        #t =tuple(offset) 
        #translation = sitk.TranslationTransform(fixedImage.GetDimension(), t)
        # Copy the translational component.
        #euler_trans = sitk.Euler3DTransform()
        #euler_trans.SetTranslation(translation.GetOffset())
        #rigid_versor_trans.SetCenter(euler_trans.GetCenter())
        #rigid_versor_trans.SetTranslation(euler_trans.GetTranslation())
        #rigid_versor_trans.SetMatrix(euler_trans.GetMatrix())
        rigid_versor_trans.SetTranslation(offset)
    else:
        euler_trans=sitk.Euler3DTransform(sitk.CenteredTransformInitializer(fixedImage,movingImage,sitk.Euler3DTransform()))
        rigid_versor_trans.SetCenter(euler_trans.GetCenter())
        rigid_versor_trans.SetTranslation(euler_trans.GetTranslation())
        rigid_versor_trans.SetMatrix(euler_trans.GetMatrix())
    
    Reg2=sitk.ImageRegistrationMethod()
    Reg2.SetInitialTransform(rigid_versor_trans,inPlace=True)
    
    # Reg2.SetMetricAsCorrelation()
    Reg2.SetMetricAsMattesMutualInformation(numberOfHistogramBins = param['numberOfBins'])
    #Reg2.SetMetricFixedMask(fixedMask)
    if mask:
      if maskType == 'moving':
        Reg2.SetMetricMovingMask(mask)
      else:
        Reg2.SetMetricFixedMask(mask)
        
    Reg2.SetInterpolator(sitk.sitkLinear)
    Reg2.SetOptimizerAsRegularStepGradientDescent(learningRate=0.2,
                                                  numberOfIterations=1500,
                                                  minStep=0.005,
                                                  relaxationFactor = 0.5,
                                                  gradientMagnitudeTolerance=1e-5,
                                                  maximumStepSizeInPhysicalUnits=0.2)
    
    Reg2.SetOptimizerScales([1.0,1.0,1.0,1.0/1000,1.0/1000,1.0/1000])
    
    smoothingSigmas=[0]
    Reg2.SetSmoothingSigmasPerLevel(smoothingSigmas)
    
    Reg2.SetMetricSamplingStrategy(Reg2.RANDOM)
    Reg2.SetMetricSamplingPercentage(param['samplingPercentage'])
    
    Reg2.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(True)
    
    shrinkFactors=[1]
    Reg2.SetShrinkFactorsPerLevel(shrinkFactors)
    
    # execute
    Reg2.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(Reg2))
    transform = Reg2.Execute(fixedImage, movingImage)
    
    # print("-------")
    # print(transform)
    # print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    # print(f" Iteration: {R.GetOptimizerIteration()}")
    # print(f" Metric value: {R.GetMetricValue()}")

    return transform
    

def registration_main(argv):

    args = []
    try:
        parser = argparse.ArgumentParser(description="Register two images .")
        parser.add_argument('fixed', metavar='FIXED_IMAGE', type=str, nargs=1,
                            help='Fixed image.')
        parser.add_argument('moving', metavar='MOVING_IMAGE', type=str, nargs=1,
                            help='Moving image')
        parser.add_argument('resampled', metavar='RESAMPLED_IMAGE', type=str, nargs=1,
                            help='Resampled')
        parser.add_argument('-m', dest='anatomLabel', default='',
                            help='Anatom label map for masking.')
        parser.add_argument('-t', dest='initialOffset', default='',
                            help='Number of control points (default: None)')

        args = parser.parse_args(argv)
        
    except Exception as e:
        print(e)
        sys.exit()

    fixedImageFile = args.fixed[0]
    movingImageFile = args.moving[0]
    resampledImageFile = args.resampled[0]
    
    param = {}
    if args.initialOffset != '':
        offset = args.initialOffset
        offset = offset.split(',')
        offset = [float(f) for f in offset]
        param['initialOffset'] = offset

    fixedImage = sitk.ReadImage (fixedImageFile, sitk.sitkFloat32)
    movingImage = sitk.ReadImage(movingImageFile, sitk.sitkFloat32)

    movingImageMasked = movingImage
    maskImage = None
    dilation = 25
    if args.anatomLabel != '':
        anatomLabel = sitk.ReadImage(args.anatomLabel, sitk.sitkInt8)
        #movingImageMasked = mask(movingImage, anatomLabel, dilation=dilation)
        #sitk.WriteImage(movingImageMasked, 'masked.nrrd')
        maskImage = createMaskFromAnatomLabel(anatomLabel, movingImage, dilation=dilation)

    #transform = registerImages(fixedImage, movingImageMasked, param)
    transform = registerImages(fixedImage, movingImage, param, mask=maskImage)
    resampledImage = resampleImage(movingImage, fixedImage, transform)
    print(transform)
    
    sitk.WriteImage(resampledImage, resampledImageFile)
    

if __name__ == "__main__":
    registration_main(sys.argv[1:])


