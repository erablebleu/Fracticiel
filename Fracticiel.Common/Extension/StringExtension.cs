
using System.IO;

namespace Fracticiel.Common.Extension
{
   public static class StringExtension
   {
      public static string GetNextAvailableFilePath(this string filePath)
      {
         if (!File.Exists(filePath))
            return filePath;

         string directory = Path.GetDirectoryName(filePath);
         string extension = Path.GetExtension(filePath);
         string fileName = Path.GetFileNameWithoutExtension(filePath);

         for(int i=1; true; i++)
         {
            string newFilePath = Path.Combine(directory, $"{fileName}_({i}).{extension}");
            if (!File.Exists(newFilePath))
               return newFilePath;
         }
      }
   }
}
