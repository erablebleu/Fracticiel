using System;
using System.Collections.Generic;
using System.Text;
using System.Windows;

namespace Fracticiel.Common
{
   public class AreaData
   {
      public Rect Area { get; set; }

      public double Resolution { get; set; }

      public long[] Data { get; set; }

      public int DataWidth { get; set; }

      public int DataHeight { get; set; }

      public long MaxValue { get; set; }

      public long LoopThreshold { get; set; }

      public AreaData()
      {

      }

      public AreaData(Rect area, double resolution, long loopThreshold)
      {
         Area = area;
         Resolution = resolution;
         LoopThreshold = loopThreshold;
         DataWidth = (int)(area.Width * Resolution);
         DataHeight = (int)(area.Height * Resolution);
         Data = new long[DataWidth * DataHeight];
      }
   }
}
