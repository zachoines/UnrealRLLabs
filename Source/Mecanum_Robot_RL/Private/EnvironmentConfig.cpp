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
        // The current node is not an object, but we still have keys left → path invalid
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
