import asyncio
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingest_doc import ingest_qa


async def test_ingest_qa():
    """Test the ingest_qa function with a sample Excel file"""
    
    # Path to your test Excel file
    file_path = "v1.xlsx"  # Change this to your actual file path
    
    print("=" * 60)
    print("Testing Q&A Ingestion")
    print("=" * 60)
    print(f"\nFile: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"\n❌ ERROR: File '{file_path}' not found!")
        print("\nPlease create an Excel file with two columns:")
        print("  - Column 1: 'Question'")
        print("  - Column 2: 'Answer'")
        return
    
    try:
        print("\n📥 Starting ingestion process...")
        
        # Call the ingest_qa function
        result = await ingest_qa(
            file_path=file_path,
            source="Test Q&A Dataset"
        )
        
        print("\n✅ SUCCESS!")
        print("-" * 60)
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        print(f"Source: {result['source']}")
        print(f"Records Inserted: {result['records_inserted']}")
        print(f"File: {result['file']}")
        print("-" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found - {e}")
        
    except ValueError as e:
        print(f"\n❌ VALIDATION ERROR: {e}")
        print("\nMake sure your Excel file has columns named 'Question' and 'Answer'")
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}")
        print(f"Details: {str(e)}")
        
        # Print more detailed error info for debugging
        import traceback
        print("\n📋 Full Traceback:")
        print("-" * 60)
        traceback.print_exc()
        print("-" * 60)


async def main():
    """Main test function"""
    
    print("\n" + "=" * 60)
    print("Q&A INGESTION TEST SCRIPT")
    print("=" * 60)
    
    # Run the ingestion test
    await test_ingest_qa()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
