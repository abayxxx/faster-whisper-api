package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
)

// Response models matching the Python API
type TranscriptSegment struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Text    string  `json:"text"`
	Speaker *string `json:"speaker,omitempty"`
}

type TranscriptionMetadata struct {
	AudioLength         float64 `json:"audio_length"`
	Language            string  `json:"language"`
	ProcessingTime      float64 `json:"processing_time"`
	DiarizationEnabled  bool    `json:"diarization_enabled"`
}

type TranscriptionResponse struct {
	Success        bool                  `json:"success"`
	Metadata       TranscriptionMetadata `json:"metadata"`
	FullTranscript string                `json:"full_transcript"`
	Segments       []TranscriptSegment   `json:"segments"`
}

type HealthResponse struct {
	Status              string `json:"status"`
	Model               string `json:"model"`
	DiarizationEnabled  bool   `json:"diarization_enabled"`
}

// TranscribeAudio sends an audio file to the transcription API
func TranscribeAudio(apiURL, audioPath, language string, enableDiarization bool, numSpeakers int) (*TranscriptionResponse, error) {
	// Open the audio file
	file, err := os.Open(audioPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open audio file: %w", err)
	}
	defer file.Close()

	// Create multipart form
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	// Add audio file
	part, err := writer.CreateFormFile("audio", filepath.Base(audioPath))
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}
	if _, err := io.Copy(part, file); err != nil {
		return nil, fmt.Errorf("failed to copy file: %w", err)
	}

	// Add optional parameters
	if language != "" {
		writer.WriteField("language", language)
	}
	if enableDiarization {
		writer.WriteField("enable_diarization", "true")
		writer.WriteField("num_speakers", fmt.Sprintf("%d", numSpeakers))
	}
	writer.WriteField("clean_audio_flag", "true")

	writer.Close()

	// Create request
	req, err := http.NewRequest("POST", apiURL+"/transcribe", body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Send request
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Parse response
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(bodyBytes))
	}

	var result TranscriptionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result, nil
}

// CheckHealth checks if the API is healthy
func CheckHealth(apiURL string) (*HealthResponse, error) {
	resp, err := http.Get(apiURL + "/health")
	if err != nil {
		return nil, fmt.Errorf("failed to check health: %w", err)
	}
	defer resp.Body.Close()

	var health HealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		return nil, fmt.Errorf("failed to decode health response: %w", err)
	}

	return &health, nil
}

func main() {
	// Example usage
	apiURL := "http://localhost:8000"

	// Check health
	health, err := CheckHealth(apiURL)
	if err != nil {
		fmt.Printf("Health check failed: %v\n", err)
		return
	}
	fmt.Printf("API Status: %s, Model: %s\n", health.Status, health.Model)

	// Transcribe audio
	audioPath := "path/to/your/audio.wav"
	result, err := TranscribeAudio(apiURL, audioPath, "en", false, 2)
	if err != nil {
		fmt.Printf("Transcription failed: %v\n", err)
		return
	}

	fmt.Printf("\nLanguage: %s\n", result.Metadata.Language)
	fmt.Printf("Duration: %.2fs\n", result.Metadata.AudioLength)
	fmt.Printf("Processing Time: %.2fs\n", result.Metadata.ProcessingTime)
	fmt.Printf("\nFull Transcript:\n%s\n", result.FullTranscript)
	
	fmt.Println("\nSegments:")
	for _, seg := range result.Segments {
		if seg.Speaker != nil {
			fmt.Printf("[%.2fs -> %.2fs] %s: %s\n", seg.Start, seg.End, *seg.Speaker, seg.Text)
		} else {
			fmt.Printf("[%.2fs -> %.2fs]: %s\n", seg.Start, seg.End, seg.Text)
		}
	}
}
