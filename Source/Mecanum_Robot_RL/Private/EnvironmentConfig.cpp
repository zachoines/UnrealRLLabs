#include "EnvironmentConfig.h"

bool UEnvironmentConfig::LoadFromFile(const FString& FilePath)
{
    FString JsonString;
    if (!FFileHelper::LoadFileToString(JsonString, *FilePath))
    {
        UE_LOG(LogTemp, Error, TEXT("UEnvironmentConfig::LoadFromFile - Failed to load file: %s"), *FilePath);
        return false;
    }

    // Parse the JSON
    TSharedPtr<FJsonObject> RootObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);
    if (!FJsonSerializer::Deserialize(Reader, RootObject) || !RootObject.IsValid())
    {
        UE_LOG(LogTemp, Error, TEXT("UEnvironmentConfig::LoadFromFile - Failed to parse JSON in file: %s"), *FilePath);
        return false;
    }

    // Wrap the root object as an FJsonValue so we can store it consistently.
    TSharedPtr<FJsonValueObject> RootValue = MakeShared<FJsonValueObject>(RootObject);
    Initialize(RootValue);

    return true;
}

UEnvironmentConfig* UEnvironmentConfig::Get(const FString& Path)
{
    if (!IsValid())
    {
        ThrowError(FString::Printf(TEXT("Cannot Get(%s); This config node is invalid."), *Path));
        return nullptr;
    }

    TArray<FString> Keys = SplitPath(Path);
    TSharedPtr<FJsonValue> SubValue = ResolvePath(InternalJsonValue, Keys, 0);
    if (!SubValue.IsValid())
    {
        ThrowError(FString::Printf(TEXT("Path not found: %s"), *Path));
        return nullptr;
    }

    // Create a new UEnvironmentConfig to wrap the sub-value
    UEnvironmentConfig* ChildConfig = NewObject<UEnvironmentConfig>(GetTransientPackage(), UEnvironmentConfig::StaticClass());
    ChildConfig->Initialize(SubValue);
    return ChildConfig;
}

FString UEnvironmentConfig::AsString() const
{
    if (!IsValid())
    {
        ThrowError(TEXT("AsString() called on invalid config node."));
        return FString();
    }

    if (InternalJsonValue->Type == EJson::String)
    {
        return InternalJsonValue->AsString();
    }
    else
    {
        ThrowError(TEXT("AsString() - This value is not a string type."));
        return FString();
    }
}

float UEnvironmentConfig::AsNumber() const
{
    if (!IsValid())
    {
        ThrowError(TEXT("AsNumber() called on invalid config node."));
        return 0.0f;
    }

    if (InternalJsonValue->Type == EJson::Number)
    {
        return static_cast<float>(InternalJsonValue->AsNumber());
    }
    else
    {
        ThrowError(TEXT("AsNumber() - This value is not a number type."));
        return 0.0f;
    }
}

int32 UEnvironmentConfig::AsInt() const
{
    if (!IsValid())
    {
        ThrowError(TEXT("AsInt() called on invalid config node."));
        return 0;
    }

    if (InternalJsonValue->Type == EJson::Number)
    {
        // Convert the double to an int
        return static_cast<int32>(InternalJsonValue->AsNumber());
    }
    else
    {
        ThrowError(TEXT("AsInt() - This value is not a number type."));
        return 0;
    }
}

bool UEnvironmentConfig::AsBool() const
{
    if (!IsValid())
    {
        ThrowError(TEXT("AsBool() called on invalid config node."));
        return false;
    }

    if (InternalJsonValue->Type == EJson::Boolean)
    {
        return InternalJsonValue->AsBool();
    }
    else
    {
        ThrowError(TEXT("AsBool() - This value is not a boolean type."));
        return false;
    }
}

TArray<float> UEnvironmentConfig::AsArrayOfNumbers() const
{
    if (!IsValid())
    {
        ThrowError(TEXT("AsArrayOfNumbers() called on invalid config node."));
        return {};
    }

    if (InternalJsonValue->Type == EJson::Array)
    {
        TArray<TSharedPtr<FJsonValue>> ArrayValues = InternalJsonValue->AsArray();
        TArray<float> Result;
        for (auto& Elem : ArrayValues)
        {
            if (!Elem.IsValid() || Elem->Type != EJson::Number)
            {
                ThrowError(TEXT("AsArrayOfNumbers() - Array element is not a number."));
                return {};
            }
            Result.Add(static_cast<float>(Elem->AsNumber()));
        }
        return Result;
    }
    else
    {
        ThrowError(TEXT("AsArrayOfNumbers() - This value is not an array."));
        return {};
    }
}

TArray<UEnvironmentConfig*> UEnvironmentConfig::AsArrayOfConfigs() const
{
    TArray<UEnvironmentConfig*> Result;

    if (!IsValid())
    {
        ThrowError(TEXT("AsArrayOfConfigs() called on invalid config node."));
        return Result;
    }

    if (InternalJsonValue->Type != EJson::Array)
    {
        ThrowError(TEXT("AsArrayOfConfigs() - This value is not an array."));
        return Result;
    }

    TArray<TSharedPtr<FJsonValue>> ArrayValues = InternalJsonValue->AsArray();
    for (int32 i = 0; i < ArrayValues.Num(); i++)
    {
        TSharedPtr<FJsonValue> Element = ArrayValues[i];
        if (!Element.IsValid())
        {
            ThrowError(FString::Printf(TEXT("AsArrayOfConfigs() - Null/invalid element at index %d"), i));
            continue;
        }

        // Create a new UEnvironmentConfig for this element
        UEnvironmentConfig* ChildConfig = NewObject<UEnvironmentConfig>(GetTransientPackage(), UEnvironmentConfig::StaticClass());
        ChildConfig->Initialize(Element);
        Result.Add(ChildConfig);
    }

    return Result;
}

FVector UEnvironmentConfig::AsFVector3() const
{
    TArray<float> Numbers = AsArrayOfNumbers();
    if (Numbers.Num() != 3)
    {
        ThrowError(FString::Printf(TEXT("AsFVector3() - Expected an array of 3 floats, got %d."), Numbers.Num()));
        return FVector::ZeroVector;
    }
    return FVector(Numbers[0], Numbers[1], Numbers[2]);
}

void UEnvironmentConfig::Initialize(const TSharedPtr<FJsonValue>& InValue)
{
    InternalJsonValue = InValue;
}

TArray<FString> UEnvironmentConfig::SplitPath(const FString& Path)
{
    TArray<FString> Out;
    Path.ParseIntoArray(Out, TEXT("/"));
    return Out;
}

TSharedPtr<FJsonValue> UEnvironmentConfig::ResolvePath(
    const TSharedPtr<FJsonValue>& CurrentValue,
    const TArray<FString>& Keys,
    int32 KeyIndex
) const
{
    if (!CurrentValue.IsValid() || KeyIndex >= Keys.Num())
    {
        return CurrentValue;
    }
    if (CurrentValue->Type != EJson::Object)
    {
        // The current node is not an object, but we still have keys left -> path invalid
        return nullptr;
    }

    // Must be an object to get the next field
    TSharedPtr<FJsonObject> CurrentObject = CurrentValue->AsObject();
    if (!CurrentObject.IsValid())
    {
        return nullptr;
    }

    FString Key = Keys[KeyIndex];
    if (!CurrentObject->HasField(Key))
    {
        return nullptr;
    }

    TSharedPtr<FJsonValue> NextValue = CurrentObject->TryGetField(Key);
    // Recurse
    return ResolvePath(NextValue, Keys, KeyIndex + 1);
}

void UEnvironmentConfig::ThrowError(const FString& Message) const
{
    // You can handle errors your own way. 
    // For simplicity, we log and ensure.
    UE_LOG(LogTemp, Error, TEXT("UEnvironmentConfig Error: %s"), *Message);
    ensureAlwaysMsgf(false, TEXT("%s"), *Message);
}

bool UEnvironmentConfig::HasPath(const FString& Path)
{
    if (!IsValid())
    {
        return false;
    }
    TArray<FString> Keys = SplitPath(Path);
    TSharedPtr<FJsonValue> SubValue = ResolvePath(InternalJsonValue, Keys, 0);
    return SubValue.IsValid();
}


float UEnvironmentConfig::GetOrDefaultNumber(const FString& Path, float DefaultVal)
{
    // If missing or invalid path => return default
    if (!HasPath(Path))
    {
        return DefaultVal;
    }

    // Otherwise, attempt to interpret
    UEnvironmentConfig* Sub = const_cast<UEnvironmentConfig*>(this)->Get(Path);
    if (!Sub || !Sub->IsValid())
    {
        return DefaultVal;
    }

    // If it's not a number, your existing code will log an error => fallback to default
    if (Sub->InternalJsonValue->Type != EJson::Number)
    {
        return DefaultVal;
    }

    return Sub->AsNumber(); // Or cast directly if you prefer
}

int32 UEnvironmentConfig::GetOrDefaultInt(const FString& Path, int32 DefaultVal)
{
    if (!HasPath(Path))
    {
        return DefaultVal;
    }

    UEnvironmentConfig* Sub = const_cast<UEnvironmentConfig*>(this)->Get(Path);
    if (!Sub || !Sub->IsValid())
    {
        return DefaultVal;
    }

    if (Sub->InternalJsonValue->Type != EJson::Number)
    {
        return DefaultVal;
    }

    return Sub->AsInt();
}

bool UEnvironmentConfig::GetOrDefaultBool(const FString& Path, bool DefaultVal)
{
    if (!HasPath(Path))
    {
        return DefaultVal;
    }

    UEnvironmentConfig* Sub = const_cast<UEnvironmentConfig*>(this)->Get(Path);
    if (!Sub || !Sub->IsValid())
    {
        return DefaultVal;
    }

    if (Sub->InternalJsonValue->Type != EJson::Boolean)
    {
        return DefaultVal;
    }

    return Sub->AsBool();
}


// Add this function definition to EnvironmentConfig.cpp
FString UEnvironmentConfig::GetOrDefaultString(const FString& Path, const FString& DefaultVal)
{
    if (!HasPath(Path))
    {
        return DefaultVal;
    }

    // Get() is not const-correct, so we might need a const_cast if 'this' is const,
    // or ensure Get() can be called from a const context if it doesn't modify state.
    // Assuming Get() is safe to call here or that this function is called on a non-const UEnvironmentConfig.
    UEnvironmentConfig* Sub = Get(Path);
    if (!Sub || !Sub->IsValid())
    {
        return DefaultVal;
    }

    if (Sub->InternalJsonValue.IsValid() && Sub->InternalJsonValue->Type != EJson::String)
    {
        UE_LOG(LogTemp, Warning, TEXT("UEnvironmentConfig::GetOrDefaultString - Path '%s' exists but is not a string. Returning default."), *Path);
        return DefaultVal;
    }
    return Sub->AsString(); // AsString() itself should handle IsValid() and type checks internally
}


TArray<float> UEnvironmentConfig::GetArrayOrDefault(const FString& Path, const TArray<float>& DefaultVal)
{
    if (!HasPath(Path))
    {
        return DefaultVal;
    }

    UEnvironmentConfig* Sub = const_cast<UEnvironmentConfig*>(this)->Get(Path);
    if (!Sub || !Sub->IsValid())
    {
        return DefaultVal;
    }

    if (Sub->InternalJsonValue->Type != EJson::Array)
    {
        return DefaultVal;
    }

    // Inside AsArrayOfNumbers(), if any element isn't numeric, it logs & returns an empty array. 
    // We'll interpret an empty array as "fallback" if you like, or just return that empty array:
    TArray<float> MaybeNums = Sub->AsArrayOfNumbers();
    if (MaybeNums.Num() == 0)
    {
        // optionally fallback to default
        return DefaultVal;
    }
    return MaybeNums;
}

FVector2D UEnvironmentConfig::GetVector2DOrDefault(const FString& Path, const FVector2D& DefaultVal)
{
    // Check path
    if (!HasPath(Path))
    {
        return DefaultVal;
    }

    UEnvironmentConfig* Sub = const_cast<UEnvironmentConfig*>(this)->Get(Path);
    if (!Sub || !Sub->IsValid())
    {
        return DefaultVal;
    }

    // Must be array, attempt to parse array of numbers
    if (Sub->InternalJsonValue->Type != EJson::Array)
    {
        return DefaultVal;
    }

    TArray<float> arr = Sub->AsArrayOfNumbers();
    if (arr.Num() != 2)
    {
        return DefaultVal;
    }

    return FVector2D(arr[0], arr[1]);
}
