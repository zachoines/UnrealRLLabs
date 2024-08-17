// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class UnrealRLLabs : ModuleRules
{
    public UnrealRLLabs(ReadOnlyTargetRules Target) : base(Target)
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
