// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class Mecanum_Robot_RL : ModuleRules
{
    public Mecanum_Robot_RL(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new string[] {
            "Core",
            "CoreUObject",
            "Engine",
            "InputCore",
            "Json",
            "JsonUtilities",
            "RenderCore", // Added for rendering support
            "RHI"         // Added for GPU interface support
        });

        PrivateDependencyModuleNames.AddRange(new string[] { });
    }
}
