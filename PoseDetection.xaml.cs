using Pose_DetectionSample.SharedCode;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media.Imaging;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using Windows.Storage.Pickers;

namespace Pose_DetectionSample;
internal sealed partial class PoseDetection : Microsoft.UI.Xaml.Controls.Page
{
    private InferenceSession? _inferenceSession;
    public PoseDetection()
    {
        this.Unloaded += (s, e) => _inferenceSession?.Dispose();
        this.InitializeComponent();
    }

    protected override async void OnNavigatedTo(Microsoft.UI.Xaml.Navigation.NavigationEventArgs e)
    {
        await InitModel(@"C:\Users\nikolame\.cache\aigallery\microsoft--dml-ai-hub-models\main\hrnet_pose\hrnet_pose.onnx");
        App.Window?.ModelLoaded();

        await DetectPose(Path.Join(Windows.ApplicationModel.Package.Current.InstalledLocation.Path, "Assets", "pose_default.png"));
    }

    private Task InitModel(string modelPath)
    {
        return Task.Run(() =>
        {
            if (_inferenceSession != null)
            {
                return;
            }

            SessionOptions sessionOptions = new();
            sessionOptions.AppendExecutionProvider_DML(DeviceUtils.GetBestDeviceId());

            _inferenceSession = new InferenceSession(modelPath, sessionOptions);
        });
    }

    private async void UploadButton_Click(object sender, RoutedEventArgs e)
    {
        var window = new Window();
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(window);

        var picker = new FileOpenPicker();
        WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);

        picker.FileTypeFilter.Add(".png");
        picker.FileTypeFilter.Add(".jpeg");
        picker.FileTypeFilter.Add(".jpg");

        picker.ViewMode = PickerViewMode.Thumbnail;

        var file = await picker.PickSingleFileAsync();
        if (file != null)
        {
            // Call function to run inference and classify image
            UploadButton.Focus(FocusState.Programmatic);
            await DetectPose(file.Path);
        }
    }

    private async Task DetectPose(string filePath)
    {
        if (!Path.Exists(filePath))
        {
            return;
        }

        Loader.IsActive = true;
        Loader.Visibility = Visibility.Visible;
        UploadButton.Visibility = Visibility.Collapsed;

        DefaultImage.Source = new BitmapImage(new Uri(filePath));

        using Bitmap image = new(filePath);

        var originalImageWidth = image.Width;
        var originalImageHeight = image.Height;

        int modelInputWidth = 256;
        int modelInputHeight = 192;

        // Resize Bitmap
        using Bitmap resizedImage = BitmapFunctions.ResizeBitmap(image, modelInputWidth, modelInputHeight);

        var predictions = await Task.Run(() =>
        {
            // Preprocessing
            Tensor<float> input = new DenseTensor<float>([1, 3, modelInputWidth, modelInputHeight]);
            input = BitmapFunctions.PreprocessBitmapWithStdDev(resizedImage, input);

            var inputMetadataName = _inferenceSession!.InputNames[0];

            // Setup inputs
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputMetadataName, input)
            };

            // Run inference
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _inferenceSession!.Run(inputs);
            var heatmaps = results[0].AsTensor<float>();

            var outputName = _inferenceSession!.OutputNames[0];
            var outputDimensions = _inferenceSession!.OutputMetadata[outputName].Dimensions;

            float outputWidth = outputDimensions[2];
            float outputHeight = outputDimensions[3];

            List<(float X, float Y)> keypointCoordinates = PoseHelper.PostProcessResults(heatmaps, originalImageWidth, originalImageHeight, outputWidth, outputHeight);
            return keypointCoordinates;
        });

        // Render predictions and create output bitmap
        using Bitmap output = PoseHelper.RenderPredictions(image, predictions, .02f);
        BitmapImage outputImage = BitmapFunctions.ConvertBitmapToBitmapImage(output);

        DispatcherQueue.TryEnqueue(() =>
        {
            DefaultImage.Source = outputImage;
            Loader.IsActive = false;
            Loader.Visibility = Visibility.Collapsed;
            UploadButton.Visibility = Visibility.Visible;
        });
    }
}