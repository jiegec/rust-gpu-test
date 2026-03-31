use anyhow::Result;
use bytemuck::cast_slice;
use pollster::FutureExt as _;
use wgpu::util::DeviceExt;
use wgpu::PollType;

async fn run() -> Result<()> {
    let n = 32u32;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| 2.0 * i as f32).collect();
    println!("a = {:?}", a);
    println!("b = {:?}", b);

    // Initialize wgpu
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await?;
    let info = adapter.get_info();
    println!("Using GPU: {} (backend: {:?})", info.name, info.backend);
    let (device, queue): (wgpu::Device, wgpu::Queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;

    // Create buffers
    let a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("a buffer"),
        contents: cast_slice(&a),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("b buffer"),
        contents: cast_slice(&b),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let c_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("c buffer"),
        size: (n * std::mem::size_of::<f32>() as u32) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Load shader
    let shader = device.create_shader_module(wgpu::include_wgsl!("compute.wgsl"));

    // Create compute pipeline
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute pipeline"),
        layout: None,
        module: &shader,
        entry_point: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: c_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch compute
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_count = (n + 63) / 64; // workgroup size 64
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }

    // Create staging buffer to read back result
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging buffer"),
        size: c_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy result to staging buffer
    encoder.copy_buffer_to_buffer(&c_buffer, 0, &staging_buffer, 0, c_buffer.size());
    queue.submit(Some(encoder.finish()));

    // Map staging buffer and read data
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device.poll(PollType::Wait {
        submission_index: None,
        timeout: None,
    })?;
    receiver.recv().unwrap()?;

    let data = buffer_slice.get_mapped_range();
    let result: &[f32] = bytemuck::cast_slice(&data);
    println!("c = {:?}", result);

    // Verification
    let expected: Vec<f32> = (0..n).map(|i| i as f32 + 2.0 * i as f32).collect();
    assert_eq!(result.len(), expected.len());
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        if (r - e).abs() > 1e-5 {
            anyhow::bail!("Mismatch at index {}: GPU {} vs CPU {}", i, r, e);
        }
    }
    println!("Verification passed!");

    // Unmap buffer
    drop(data);
    staging_buffer.unmap();

    Ok(())
}

fn main() -> Result<()> {
    run().block_on()
}
