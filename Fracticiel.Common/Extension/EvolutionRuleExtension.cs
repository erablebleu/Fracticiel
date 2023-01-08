using Fracticiel.Common.Enum;
using System;
using System.Collections.Generic;
using System.Text;

namespace Fracticiel.Common.Extension
{
   public static class EvolutionRuleExtension
   {
      public static byte GetValue(this EvolutionRule evolutionRule, double val)//0<=val<=1
      {
         switch (evolutionRule)
         {
            case EvolutionRule.Logarithmic: return (byte)(50 * Math.Log(1 + 164 * val));
            case EvolutionRule.Sinus: return (byte)(255 * Math.Sin(Math.PI * val / 2));
            case EvolutionRule.SQRT: return (byte)(255 * Math.Sqrt(val));
            case EvolutionRule.Square: return (byte)(255 * val * val);
            case EvolutionRule.Exponential:
            case EvolutionRule.Linear:
            default: return (byte)(255 * val);
         }
      }
   }
}
