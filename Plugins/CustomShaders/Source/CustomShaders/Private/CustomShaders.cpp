#include "CustomShaders.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"

#define LOCTEXT_NAMESPACE "FCustomShadersModule"

void FCustomShadersModule::StartupModule()
{
	FString ShaderDir = FPaths::Combine(IPluginManager::Get().FindPlugin("CustomShaders")->GetBaseDir(), TEXT("Shaders"));
	AddShaderSourceDirectoryMapping(TEXT("/Plugins/CustomShaders"), ShaderDir);
}

void FCustomShadersModule::ShutdownModule()
{

}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FCustomShadersModule, CustomShaders)
