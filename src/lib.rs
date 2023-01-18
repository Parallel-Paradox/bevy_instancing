use std::marker::PhantomData;
use std::hash::Hash;

use bevy::{
    prelude::*,
    render::{
        extract_component::{ExtractComponentPlugin, ExtractComponent}, RenderApp, RenderStage,
        render_asset::{PrepareAssetLabel, RenderAssets},
        render_phase::{
            SetItemPipeline, PhaseItem, RenderCommand, TrackedRenderPass, RenderCommandResult, AddRenderCommand,
            DrawFunctions, RenderPhase
        },
        mesh::{GpuBufferInfo, MeshVertexBufferLayout},
        render_resource::{
            Buffer, BufferInitDescriptor, BufferUsages, SpecializedMeshPipeline, SpecializedMeshPipelineError,
            RenderPipelineDescriptor, VertexBufferLayout, SpecializedMeshPipelines, PipelineCache
        },
        renderer::RenderDevice, view::{ExtractedView, VisibleEntities}
    },
    ecs::{query::QueryItem, system::{lifetimeless::{SRes, Read}, SystemParamItem}},
    pbr::{
        ExtractedMaterials, RenderMaterials, extract_materials, prepare_materials, SetMeshViewBindGroup,
        SetMaterialBindGroup, SetMeshBindGroup, MaterialPipeline, MaterialPipelineKey, MeshUniform, MeshPipelineKey
    },
    core_pipeline::{core_3d::{Transparent3d, Opaque3d, AlphaMask3d}, tonemapping::Tonemapping}
};
use bytemuck::{Pod, Zeroable};

pub struct InstanceMaterialPlugin<M: Material, D: InstanceData>(PhantomData<M>, PhantomData<D>);

impl<M: Material, D: InstanceData> Plugin for InstanceMaterialPlugin<M, D>
where
    M::Data: PartialEq + Eq + Hash + Clone
{
    fn build(&self, app: &mut App) {
        app.add_asset::<M>()
            .add_plugin(ExtractComponentPlugin::<Handle<M>>::extract_visible())
            .add_plugin(ExtractComponentPlugin::<InstanceDataVec<D>>::extract_visible());
        
        app.sub_app_mut(RenderApp)
            .init_resource::<ExtractedMaterials<M>>()
            .init_resource::<RenderMaterials<M>>()
            .add_system_to_stage(RenderStage::Extract, extract_materials::<M>)
            .add_system_to_stage(
                RenderStage::Prepare,
                prepare_materials::<M>.after(PrepareAssetLabel::PreAssetPrepare),
            )
            .add_system_to_stage(RenderStage::Prepare, prepare_instance_buffers::<D>)
            
            .add_render_command::<Transparent3d, DrawInstancedMaterial<M>>()
            .add_render_command::<Opaque3d, DrawInstancedMaterial<M>>()
            .add_render_command::<AlphaMask3d, DrawInstancedMaterial<M>>()
            .init_resource::<InstanceMaterialPipeline<M, D>>()
            .init_resource::<SpecializedMeshPipelines<InstanceMaterialPipeline<M, D>>>()
            .add_system_to_stage(RenderStage::Queue, queue_instance_material_meshes::<M, D>);
    }
}

#[derive(Component, Deref)]
pub struct InstanceDataVec<D: InstanceData>(Vec<D>);

impl<D: InstanceData> ExtractComponent for InstanceDataVec<D> {
    type Query = &'static InstanceDataVec<D>;
    type Filter = ();
    type Out = Self;

    fn extract_component(item: QueryItem<'_, Self::Query>) -> Option<Self> {
        Some(InstanceDataVec::<D>(item.0.clone()))
    }
}

pub trait InstanceData : Clone + Copy + Send + Sync + Pod + Zeroable {
    fn buffer_layout() -> VertexBufferLayout;
    fn buffer_label() -> &'static str { std::any::type_name::<Self>() }
}

#[derive(Component)]
pub struct InstanceBuffer {
    buffer: Buffer,
    length: usize,
}

pub fn prepare_instance_buffers<D: InstanceData>(
    mut commands: Commands,
    query: Query<(Entity, &InstanceDataVec<D>)>,
    render_device: Res<RenderDevice>,
) {
    for (entity, instance_data) in &query {
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some(D::buffer_label()),
            contents: bytemuck::cast_slice(instance_data.as_slice()),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });
        commands.entity(entity).insert(InstanceBuffer {
            buffer,
            length: instance_data.len(),
        });
    }
}

type DrawInstancedMaterial<M> = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMaterialBindGroup<M, 1>,
    SetMeshBindGroup<2>,
    DrawMeshInstanced,
);

pub struct DrawMeshInstanced;

impl<P: PhaseItem> RenderCommand<P> for DrawMeshInstanced {
    type Param = SRes<RenderAssets<Mesh>>;
    type ViewWorldQuery = ();
    type ItemWorldQuery = (Read<Handle<Mesh>>, Read<InstanceBuffer>);

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        (mesh_handle, instance_buffer): (&'w Handle<Mesh>, &'w InstanceBuffer),
        meshes: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let gpu_mesh = match meshes.into_inner().get(mesh_handle) {
            Some(gpu_mesh) => gpu_mesh,
            None => return RenderCommandResult::Failure,
        };

        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, instance_buffer.buffer.slice(..));

        match &gpu_mesh.buffer_info {
            GpuBufferInfo::Indexed {
                buffer,
                index_format,
                count,
            } => {
                pass.set_index_buffer(buffer.slice(..), 0, *index_format);
                pass.draw_indexed(0..*count, 0, 0..instance_buffer.length as u32);
            }
            GpuBufferInfo::NonIndexed { vertex_count } => {
                pass.draw(0..*vertex_count, 0..instance_buffer.length as u32);
            }
        }
        RenderCommandResult::Success
    }
}

fn queue_instance_material_meshes<M: Material, D: InstanceData>(
    opaque_draw_functions: Res<DrawFunctions<Opaque3d>>,
    alpha_mask_draw_functions: Res<DrawFunctions<AlphaMask3d>>,
    transparent_draw_functions: Res<DrawFunctions<Transparent3d>>,
    instance_material_pipeline: Res<InstanceMaterialPipeline<M, D>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<InstanceMaterialPipeline<M, D>>>,
    pipeline_cache: Res<PipelineCache>,
    msaa: Res<Msaa>,
    render_meshes: Res<RenderAssets<Mesh>>,
    render_materials: Res<RenderMaterials<M>>,
    instance_material_meshes: Query<(&Handle<M>, &Handle<Mesh>, &MeshUniform), With<InstanceDataVec<D>>>,
    mut views: Query<(
        &ExtractedView,
        &VisibleEntities,
        Option<&Tonemapping>,
        &mut RenderPhase<Opaque3d>,
        &mut RenderPhase<AlphaMask3d>,
        &mut RenderPhase<Transparent3d>,
    )>,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    for (
        view,
        visible_entities,
        tonemapping,
        mut opaque_phase,
        mut alpha_mask_phase,
        mut transparent_phase,
    ) in &mut views {
        let draw_opaque_pbr = opaque_draw_functions.read().id::<DrawInstancedMaterial<M>>();
        let draw_alpha_mask_pbr = alpha_mask_draw_functions.read().id::<DrawInstancedMaterial<M>>();
        let draw_transparent_pbr = transparent_draw_functions.read().id::<DrawInstancedMaterial<M>>();

        let mut view_key =
            MeshPipelineKey::from_msaa_samples(msaa.samples) | MeshPipelineKey::from_hdr(view.hdr);
        
        if let Some(Tonemapping::Enabled { deband_dither }) = tonemapping {
            if !view.hdr {
                view_key |= MeshPipelineKey::TONEMAP_IN_SHADER;

                if *deband_dither {
                    view_key |= MeshPipelineKey::DEBAND_DITHER;
                }
            }
        }
        let rangefinder = view.rangefinder3d();
        
        for visible_entity in &visible_entities.entities {
            let (
                material_handle,
                mesh_handle,
                mesh_uniform
            ) = match instance_material_meshes.get(*visible_entity) {
                Ok(mesh) => mesh,
                Err(_) => return,
            };
            let material = match render_materials.get(material_handle) {
                Some(material) => material,
                None => return,
            };
            let mesh = match render_meshes.get(mesh_handle) {
                Some(mesh) => mesh,
                None => return,
            };

            let mut mesh_key =
                MeshPipelineKey::from_primitive_topology(mesh.primitive_topology) | view_key;
            let alpha_mode = material.properties.alpha_mode;
            if let AlphaMode::Blend = alpha_mode {
                mesh_key |= MeshPipelineKey::TRANSPARENT_MAIN_PASS;
            }

            let pipeline_id = pipelines.specialize(
                &pipeline_cache,
                &instance_material_pipeline,
                MaterialPipelineKey {
                    mesh_key,
                    bind_group_data: material.key.clone(),
                },
                &mesh.layout,
            );
            let pipeline_id = match pipeline_id {
                Ok(id) => id,
                Err(err) => {
                    error!("{}", err);
                    continue;
                }
            };

            let distance = rangefinder.distance(&mesh_uniform.transform) + material.properties.depth_bias;
            match alpha_mode {
                AlphaMode::Opaque => {
                    opaque_phase.add(Opaque3d {
                        entity: *visible_entity,
                        draw_function: draw_opaque_pbr,
                        pipeline: pipeline_id,
                        distance,
                    });
                }
                AlphaMode::Mask(_) => {
                    alpha_mask_phase.add(AlphaMask3d {
                        entity: *visible_entity,
                        draw_function: draw_alpha_mask_pbr,
                        pipeline: pipeline_id,
                        distance,
                    });
                }
                AlphaMode::Blend => {
                    transparent_phase.add(Transparent3d {
                        entity: *visible_entity,
                        draw_function: draw_transparent_pbr,
                        pipeline: pipeline_id,
                        distance,
                    });
                }
            }
        }
    }
}

#[derive(Resource, Clone)]
pub struct InstanceMaterialPipeline<M: Material, D: InstanceData> {
    pub material_pipeline: MaterialPipeline<M>,
    _data: PhantomData<D>,
}

impl<M: Material, D: InstanceData> FromWorld for InstanceMaterialPipeline<M, D> {
    fn from_world(world: &mut World) -> Self {
        InstanceMaterialPipeline {
            material_pipeline: MaterialPipeline::from_world(world),
            _data: Default::default()
        }
    }
}

impl<M: Material, D: InstanceData> SpecializedMeshPipeline for InstanceMaterialPipeline<M, D>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    type Key = MaterialPipelineKey<M>;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayout,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.material_pipeline.specialize(key, layout)?;
        descriptor.vertex.buffers.push(D::buffer_layout());
        Ok(descriptor)
    }
}
