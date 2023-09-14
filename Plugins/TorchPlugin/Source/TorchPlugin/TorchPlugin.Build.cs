using UnrealBuildTool;
using System.IO;
using System;

public class TorchPlugin : ModuleRules
{
    public TorchPlugin(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        // Delay-load the DLL
        //PublicDelayLoadDLLs.Add("$(PluginDir)/Binaries/Win64/TorchWrapper.dll");

        // Ensure the DLL is staged along with the executable
        // RuntimeDependencies.Add("$(PluginDir)/Binaries/Win64/TorchWrapper.dll");

        // Add the .lib file for linking
        // PublicAdditionalLibraries.Add("$(PluginDir)/Binaries/Win64/TorchWrapper.lib");

        bUseRTTI = true;
        bEnableExceptions = true;

        PublicDefinitions.Add("TORCHPLUGIN_EXPORTS=1");

        PublicIncludePaths.AddRange(
            new string[] {
				// ... add public include paths required here ...
			}
            );

        PrivateIncludePaths.AddRange(
            new string[] {
				// ... add other private include paths required here ...
			}
            );

        PublicDependencyModuleNames.AddRange(
            new string[]
            {
                "Core",
				// ... add other public dependencies that you statically link with here ...
			}
            );

        PrivateDependencyModuleNames.AddRange(
            new string[]
            {
                "CoreUObject",
                "Engine",
                "Slate",
                "SlateCore",
				// ... add private dependencies that you statically link with here ...	
			}
            );

        DynamicallyLoadedModuleNames.AddRange(
            new string[]
            {
				// ... add any modules that your module loads dynamically here ...
			}
            );
    }
}
