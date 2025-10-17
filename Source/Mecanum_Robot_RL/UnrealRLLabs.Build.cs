// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using UnrealBuildTool.Rules;

public class UnrealRLLabs : ModuleRules
{
    public UnrealRLLabs(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDefinitions.Add("UNREALRLLABS_ENABLE_CHAOS_CONTACT_FILTER=1");
        PublicDependencyModuleNames.AddRange(new string[] {
            "Core",
            "CoreUObject",
            "Engine",
            "InputCore",
            "Json",
            "JsonUtilities",
            "RenderCore", // Added for rendering support
            "RHI",         // Added for GPU interface support
            "ProceduralMeshComponent",
            "Niagara",
            "CustomShaders",
            // Physics/Chaos for contact filtering support
            "PhysicsCore",
            "Chaos",
            "ChaosSolverEngine"
        });

        PrivateDependencyModuleNames.AddRange(new string[] { });
    }
}
