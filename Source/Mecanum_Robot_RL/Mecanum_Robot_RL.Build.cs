// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class Mecanum_Robot_RL : ModuleRules
{
    public Mecanum_Robot_RL(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "TorchPlugin" });

        PrivateDependencyModuleNames.AddRange(new string[] { });

        // Ensure the DLL is staged along with the executable
        // RuntimeDependencies.Add("TorchWrapper.dll");

        // Add the .lib file for linking
        // PublicAdditionalLibraries.Add("TorchWrapper.dll");


        // Uncomment if you are using Slate UI
        // PrivateDependencyModuleNames.AddRange(new string[] { "Slate", "SlateCore" });

        // Uncomment if you are using online features
        // PrivateDependencyModuleNames.Add("OnlineSubsystem");

        // To include OnlineSubsystemSteam, add it to the plugins section in your uproject file with the Enabled attribute set to true
    }
}
