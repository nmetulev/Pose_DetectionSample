using Microsoft.UI.Xaml;

namespace Pose_DetectionSample
{
    public sealed partial class MainWindow : Window
    {
        public MainWindow()
        {
            this.InitializeComponent();
            this.RootFrame.Loaded += (sender, args) =>
            {
                RootFrame.Navigate(typeof(PoseDetection));
            };
        }

        internal void ModelLoaded()
        {
            ProgressRingGrid.Visibility = Visibility.Collapsed;
        }
    }
}
