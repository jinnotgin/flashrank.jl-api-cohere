using Genie
using Genie.Router
using Genie.Requests
using Genie.Renderer.Json
using FlashRank
using JSON3
using Logging
using UUIDs

# Set up logging
logger = SimpleLogger(stdout, Logging.Info)
global_logger(logger)

# Initialize the ranker
const ranker = RankerModel() # (default) ms-marco-TinyBERT-L-2-v2 (alias :tiny)
# const ranker = RankerModel(:mini4) # ms-marco-MiniLM-L-4-v2 ONNX (alias :mini4)
# const ranker = RankerModel(:mini6) # ms-marco-MiniLM-L-6-v2 ONNX (alias :mini6)
# const ranker = RankerModel(:mini12) # ms-marco-MiniLM-L-12-v2 (alias :mini or mini12)

# Define the request and response structures
struct RankRequest
    model::String
    query::String
    documents::Vector{Any}  # Accommodates both String and Dict
    top_n::Union{Nothing, Int}
    rank_fields::Union{Nothing, Vector{String}}
    return_documents::Bool
    max_chunks_per_doc::Union{Nothing, Int}
end

struct RankResponseItem
    document::Union{Nothing, Dict{String, Any}}
    index::Int
    relevance_score::Float64
end

struct RankResponseMeta
    api_version::Dict{String, Any}
    billed_units::Dict{String, Int}
    tokens::Dict{String, Int}
    warnings::Vector{String}
end

struct RankResponse
    id::String
    results::Vector{RankResponseItem}
    meta::RankResponseMeta
end

# Helper function to extract text from various document formats
function extract_text(doc::Any, rank_fields::Union{Nothing, Vector{String}})
	# @info "Extracting text from document" typeof(doc) rank_fields
	if isa(doc, String)
			return doc
	elseif isa(doc, JSON3.Object)
			if isnothing(rank_fields)
					return get(doc, "text", "")
			else
					extracted_text = join([string(get(doc, String(field), "")) for field in rank_fields], " ")
					return extracted_text
			end
	else
			throw(ArgumentError("Unsupported document type: $(typeof(doc))"))
	end
end

# Helper function to generate response
function generate_response(request::RankRequest)
	@info "Generating response" request

    # Extract passages from the request
    passages = [extract_text(doc, request.rank_fields) for doc in request.documents]
    @info "Extracted passages" num_passages=length(passages)
    
    # Perform ranking
    @info "Performing ranking"
    result = FlashRank.rank(ranker, request.query, passages)
    
    # Prepare the results
    @info "Preparing results"
    final_result = [
        RankResponseItem(
            prepare_document_output(request.documents[result.positions[i]], request.return_documents),
            result.positions[i] - 1,  # Convert 1-based index to 0-based
            result.scores[i]
        )
        for i in 1:length(result.scores)
    ]
    
    # Apply top_n filter if specified
    if !isnothing(request.top_n)
        @info "Applying top_n filter" top_n=request.top_n
        final_result = final_result[1:min(request.top_n, length(final_result))]
    end
    
    # Create the response with meta information
    return RankResponse(
        string(uuid4()),
        final_result,
        RankResponseMeta(
            Dict("version" => "1.0", "is_deprecated" => false, "is_experimental" => false),
            Dict("input_tokens" => 0, "output_tokens" => 0, "search_units" => 0, "classifications" => 0),
            Dict("input_tokens" => 0, "output_tokens" => 0),
            String[]
        )
    )
end

# Helper function to prepare document output
function prepare_document_output(doc::Any, return_documents::Bool)
	if !return_documents
			return nothing
	elseif isa(doc, String)
			return Dict("text" => doc)
	elseif isa(doc, JSON3.Object)
			return Dict(String(k) => v for (k, v) in pairs(doc))
	else
			throw(ArgumentError("Unsupported document type: $(typeof(doc))"))
	end
end


# Define the rerank handler function
function rerank_handler()
	try
			# Parse the request payload using JSON3
			req_data = JSON3.read(rawpayload())
			
			# Construct the RankRequest object
			request = RankRequest(
					req_data.model,
					req_data.query,
					req_data.documents,
					get(req_data, :top_n, nothing),
					get(req_data, :rank_fields, nothing),
					get(req_data, :return_documents, true),
					get(req_data, :max_chunks_per_doc, nothing)
			)
			
			# Log the incoming request
			@info "Received rerank request" request
			
			# Generate the response
			response = generate_response(request)
			
			# Log the outgoing response
			@info "Sending rerank response" response
			
			# Return the JSON response
			return json(response)
	catch e
			# Log any errors with more detail
			@error "Error processing request" exception=(e, catch_backtrace())
			println("Error details: ", e)
			
			# Return a more detailed error response
			return json(Dict(
					"error" => "Internal server error",
					"message" => string(e),
					"type" => string(typeof(e))
			), status=500)
	end
end

# Define the routes
route("/v1/rerank", rerank_handler, method="POST")
route("/rerank", rerank_handler, method="POST")

# Start the server
up(5971, "0.0.0.0", async=false)
