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
    /**
     * Returns true if the given slash-delimited path (e.g. "environment/shape/agent")
     * exists in the JSON hierarchy. Otherwise false.
     */
    UFUNCTION(BlueprintCallable, Category = "Config")
    bool HasPath(const FString& Path);

    /**
     * Loads a JSON file and stores its root object internally.
     * Throws an error (UE_LOG + returns false) if file not found or parse error.
     */
    UFUNCTION(BlueprintCallable, Category = "Config")
    bool LoadFromFile(const FString& FilePath);

    /**
     * Look up a nested path like "environment/params/ObjectSize" and return a *subtree*
     * as a new UEnvironmentConfig. If any part is missing or not an object/array, we throw an error.
     */
    UFUNCTION(BlueprintCallable, Category = "Config")
    UEnvironmentConfig* Get(const FString& Path);

    /**
     * Interpret the current node as a string. If it's not a string, we throw an error.
     */
    UFUNCTION(BlueprintCallable, Category = "Config")
    FString AsString() const;

    /**
     * Interpret the current node as a number (float). If it's not numeric, we throw an error.
     */
    UFUNCTION(BlueprintCallable, Category = "Config")
    float AsNumber() const;

    /**
     * Interpret the current node as an integer. If it's not numeric, we throw an error.
     */
    UFUNCTION(BlueprintCallable, Category = "Config")
    int32 AsInt() const;

    /**
     * Interpret the current node as a bool. If it's not a bool, we throw an error.
     */
    UFUNCTION(BlueprintCallable, Category = "Config")
    bool AsBool() const;

    /**
     * Interpret the current node as an array of numeric values (float).
     * If it's not an array of numbers, we throw an error.
     */
    UFUNCTION(BlueprintCallable, Category = "Config")
    TArray<float> AsArrayOfNumbers() const;

    /**  
     * Returns an array of sub-configs, if this node is an array of objects. 
     */
    UFUNCTION(BlueprintCallable, Category = "Config")
    TArray<UEnvironmentConfig*> AsArrayOfConfigs() const;

    /**
     * Optionally: Convert array of 3 floats to a FVector
     * (Throws if the array isn’t exactly length 3 or not numeric.)
     */
    UFUNCTION(BlueprintCallable, Category = "Config")
    FVector AsFVector3() const;

    /**
     * Return whether this node is valid.
     */
    bool IsValid() const { return InternalJsonValue.IsValid(); }

    /**
     * Called internally to wrap a sub-value in a new UEnvironmentConfig object.
     */
    void Initialize(const TSharedPtr<FJsonValue>& InValue);

private:
    /**
     * The underlying JSON node we represent.
     * It could be an object, array, string, bool, number, etc.
     */
    TSharedPtr<FJsonValue> InternalJsonValue;

    /**
     * Helper for splitting paths like "environment/params/ObjectSize" into ["environment","params","ObjectSize"].
     */
    static TArray<FString> SplitPath(const FString& Path);

    /**
     * Internal function used by Get(...) to descend into JSON objects and arrays.
     */
    TSharedPtr<FJsonValue> ResolvePath(const TSharedPtr<FJsonValue>& CurrentValue, const TArray<FString>& Keys, int32 KeyIndex) const;

    /**
     * Helper to throw an error with a clear message.
     */
    void ThrowError(const FString& Message) const;
};
