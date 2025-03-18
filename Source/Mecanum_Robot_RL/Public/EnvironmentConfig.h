#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonReader.h"
#include "EnvironmentConfig.generated.h"

/**
 * A flexible config class that wraps a JSON tree and allows path-based lookups.
 */
UCLASS(BlueprintType)
class UNREALRLLABS_API UEnvironmentConfig : public UObject
{
    GENERATED_BODY()

public:
    // Existing API (same as your code)
    UFUNCTION(BlueprintCallable, Category = "Config")
    bool LoadFromFile(const FString& FilePath);

    UFUNCTION(BlueprintCallable, Category = "Config")
    UEnvironmentConfig* Get(const FString& Path);

    UFUNCTION(BlueprintCallable, Category = "Config")
    bool HasPath(const FString& Path);

    UFUNCTION(BlueprintCallable, Category = "Config")
    FString AsString() const;

    UFUNCTION(BlueprintCallable, Category = "Config")
    float AsNumber() const;

    UFUNCTION(BlueprintCallable, Category = "Config")
    int32 AsInt() const;

    UFUNCTION(BlueprintCallable, Category = "Config")
    bool AsBool() const;

    UFUNCTION(BlueprintCallable, Category = "Config")
    TArray<float> AsArrayOfNumbers() const;

    UFUNCTION(BlueprintCallable, Category = "Config")
    TArray<UEnvironmentConfig*> AsArrayOfConfigs() const;

    UFUNCTION(BlueprintCallable, Category = "Config")
    FVector AsFVector3() const;

    bool IsValid() const { return InternalJsonValue.IsValid(); }

    // -----------------------------
    // NEW: "GetOrDefault" Methods
    // -----------------------------
    UFUNCTION(BlueprintCallable, Category = "Config")
    float GetOrDefaultNumber(const FString& Path, float DefaultVal);

    UFUNCTION(BlueprintCallable, Category = "Config")
    int32 GetOrDefaultInt(const FString& Path, int32 DefaultVal);

    UFUNCTION(BlueprintCallable, Category = "Config")
    bool GetOrDefaultBool(const FString& Path, bool DefaultVal);

    /**
     * For numeric arrays. If the path is missing or invalid, returns DefaultVal.
     */
    UFUNCTION(BlueprintCallable, Category = "Config")
    TArray<float> GetArrayOrDefault(const FString& Path, const TArray<float>& DefaultVal);

    /**
     * For reading a 2-float array into an FVector2D.
     * If invalid or the array isn't exactly 2 floats, returns DefaultVal.
     */
    UFUNCTION(BlueprintCallable, Category = "Config")
    FVector2D GetVector2DOrDefault(const FString& Path, const FVector2D& DefaultVal);

    // Called internally to wrap a sub-value in a new UEnvironmentConfig object.
    void Initialize(const TSharedPtr<FJsonValue>& InValue);

private:
    // Internal JSON node we represent
    TSharedPtr<FJsonValue> InternalJsonValue;

    static TArray<FString> SplitPath(const FString& Path);
    TSharedPtr<FJsonValue> ResolvePath(
        const TSharedPtr<FJsonValue>& CurrentValue,
        const TArray<FString>& Keys,
        int32 KeyIndex) const;

    void ThrowError(const FString& Message) const;
};
